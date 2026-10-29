#!/usr/bin/env python3
"""
MSIBI optimization run script (cluster-friendly, no CLI args).

Concessions made (minimal, but important for cluster):
- No notebook magics.
- No argparse / no flags.
- All paths remain hard-coded exactly as you had them.
- State initialization is unchanged (same constructors/values).
- You can still choose states by commenting/uncommenting optimizer.add_state(...)
- Wrapped everything in main() so the file is safe to import and behaves predictably on clusters.
- Removed duplicate helper function re-definitions (deg/key/infer_rcut_nbins) by keeping one copy.
- Kept your print() statements (your SLURM stdout->run.log will capture them).
"""

import warnings
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import gsd.hoomd
import hoomd
import msibi.forces
from msibi.utils.corrections import linear, pair_corrections, bonded_corrections
from msibi import State, Bond, Angle, Pair, MSIBI
import json
import os
import shutil

warnings.filterwarnings("ignore")

PRETRAINED_DIR = Path("/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI_GPU_RUNS/run1/msibi_outputs")
FFA_PRETRAINED_DIR = Path("/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI_GPU_RUNS/FFA_PAIR_OPT_CYCLES/run1/msibi_outputs")


PAIR_DIR = Path("/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/cg")
EXTRA_DIR = Path("/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/eosFF_6/0")
INIT_DIR = Path("/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/eosFF_6/0")

RINGS = ("ring1", "ring2", "ring3", "ring4")


D_u = 6.0   # Angstrom
E_u = 0.1   # kcal/mol

def deg(x: float) -> float:
    return np.deg2rad(x)

def bond_r0(x_ang: float) -> float:
    return x_ang / D_u

def bond_k(k_phys: float) -> float:
    return k_phys * (D_u ** 2) / E_u

def angle_k(k_phys: float) -> float:
    return k_phys / E_u

def get_bond_angle_types(gsd_path: str):
    with gsd.hoomd.open(gsd_path, "r") as traj:
        snap = traj[0]
        bond_types = list(snap.bonds.types)
        angle_types = list(snap.angles.types)
    return bond_types, angle_types


def build_bond_dict(bond_types, bond_forces):
    bond_dict = {}
    for bond_name in bond_types:
        type1, type2 = bond_name.split("-")

        key1 = "bond_" + "_".join([type1, type2])
        key2 = "bond_" + "_".join([type2, type1])

        if key1 in bond_forces:
            bond_dict[bond_name] = bond_forces[key1]
        elif key2 in bond_forces:
            bond_dict[bond_name] = bond_forces[key2]
        else:
            print(f"{bond_name} not found in bonds.json")

    return bond_dict


def build_angle_dict(angle_types, angle_forces):
    angle_dict = {}
    for angle_name in angle_types:
        type1, type2, type3 = angle_name.split("-")

        key1 = "angle_" + "_".join([type1, type2, type3])
        key2 = "angle_" + "_".join([type3, type2, type1])

        if key1 in angle_forces:
            angle_dict[angle_name] = angle_forces[key1]
        elif key2 in angle_forces:
            angle_dict[angle_name] = angle_forces[key2]
        else:
            print(f"{angle_name} not found in angles.json")

    return angle_dict


def add_bonds_from_dict(optimizer, bond_dict):
    for bond_name, info in bond_dict.items():
        bond_obj = Bond(
            type1=info["type1"],
            type2=info["type2"],
            optimize=False
        )
        bond_obj.set_harmonic(
            k=bond_k(info["k"]),
            r0=bond_r0(info["r0"])
        )
        optimizer.add_force(bond_obj)


def add_angles_from_dict(optimizer, angle_dict):
    for angle_name, info in angle_dict.items():
        angle_obj = Angle(
            type1=info["type1"],
            type2=info["type2"],
            type3=info["type3"],
            optimize=False
        )
        angle_obj.set_harmonic(
            k=angle_k(info["k"]),
            t0=np.deg2rad(info["t0_deg"])
        )
        optimizer.add_force(angle_obj)

def key(t1: str, t2: str):
    return tuple(sorted((t1, t2)))


def parse_types(filename: str):
    stem = Path(filename).stem
    type1, type2 = stem.split("-")
    return type1, type2


def infer_rcut_nbins(filepath: Path):
    if filepath.suffix == ".csv":
        df = pd.read_csv(filepath)
        r = df["x"].to_numpy()
        return float(r.max()), int(len(r))

    data = np.loadtxt(filepath, comments=["#", "@"])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    r = data[:, 0]
    return float(r.max()), int(len(r))

def hoomd_table_to_msibi_csv(txt_path: Path, out_csv: Path) -> Path:
    arr = np.loadtxt(txt_path, comments=["#", "@"])
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise ValueError(f"{txt_path} must have at least 2 columns: r U")

    r = arr[:, 0]
    U = arr[:, 1]

    df = pd.DataFrame({"x": r, "potential": U})

    if arr.shape[1] >= 3:
        F = arr[:, 2]
        df["force"] = F

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def gsd_types(gsd_path: str) -> set[str]:
    t = gsd.hoomd.open(gsd_path, "r")
    return set(t[0].particles.types)


def compute_safe_rmax(states):
    values = []
    for s in states:
        t = gsd.hoomd.open(s.traj_file, "r")
        Lx, Ly, Lz = t[0].configuration.box[:3]
        values.append(0.49 * min(Lx, Ly, Lz))
    return min(values)


def types_in_state(state):
    t = gsd.hoomd.open(state.traj_file, "r")
    return set(t[0].particles.types)


def compute_types_sets(states):
    per_state = [types_in_state(s) for s in states]
    types_any = set().union(*per_state) if per_state else set()
    types_all = set.intersection(*per_state) if per_state else set()
    return per_state, types_any, types_all


def find_table_file(pair_dir: Path, t1: str, t2: str) -> Path:
    f1 = pair_dir / f"{t1}-{t2}.txt"
    f2 = pair_dir / f"{t2}-{t1}.txt"
    if f1.exists():
        return f1
    if f2.exists():
        return f2
    raise FileNotFoundError(
        f"Missing initial guess table for {t1}-{t2}: expected {f1.name} or {f2.name} in {pair_dir}"
    )


def expand_ring_pairs(t1: str, t2: str):
    return [(t1, t2)]

STAGE_1 = {
    key("cterm", "sticky"),
    key("chead", "sticky"),
    key("ctail", "sticky"),
    key("chme", "sticky"),
    key("ring", "sticky"),
}

STAGE_2 = {
    key("ester", "chead"),
    key("ester", "ctail"),
    key("ester", "chme"),
    key("ester", "cterm"),
    key("ester", "ring"),
}

STAGE_3 = {
    key("taild", "ctail"),
    key("taild", "cterm"),
    key("taild", "chme"),
    key("taild", "chead"),
    key("taild", "ring"),
}

STAGE_4 = {
    key("tail2", "cterm"),
    key("tail2", "chead"),
    key("tail2", "ctail"),
    key("tail2", "chme"),
    key("tail2", "ring"),
}

ALL_STAGE_PAIRS = STAGE_1 | STAGE_2 | STAGE_3 | STAGE_4

CURRENT_STAGE = 1

# Unoptimises pairs in a stage for all states, add as key("X","Y"),
MANUAL_FORCE_FIXED = {
    }


STAGE_ORDER = {
    1: STAGE_1,
    2: STAGE_2,
    3: STAGE_3,
    4: STAGE_4,
}

def expand_pair_ignore_states(raw_ignore_map):
    expanded = {}
    for (a, b), state_names in raw_ignore_map.items():
        for x, y in expand_ring_pairs(a, b):
            expanded[key(x, y)] = set(state_names)
    return expanded

ALL_STATES = {
    "bulk_EOS_CHOL",
    "EOS_CHOL_1_1_bilayer",
    "EOS_CHOL_1_2_bilayer",
    "EOS_LPP",
}

# Pairs ignores states they arent properly defined in.
# In the dict below, we ignore ALL States besides the ones inside the { }.

RAW_PAIR_IGNORE_STATES = {
    # Stage 1
    key("cterm", "sticky"): {
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
    },

    key("chead", "sticky"): set(),
    key("ctail", "sticky"): set(),
    key("chme", "sticky"): set(),
    key("ring", "sticky"): set(),

    # Stage 2
    key("ester", "chead"): set(),
    key("ester", "ctail"): set(),
    key("ester", "chme"): set(),
    key("ester", "cterm"): set(),
    key("ester", "ring"): set(),

    # Stage 3: not optimized against CHOL-EOS bilayers
    key("taild", "ctail"): {
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
    },
    key("taild", "cterm"): {
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
    },
    key("taild", "chme"): {
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
    },
    key("taild", "chead"): {
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
    },
    key("taild", "ring"): {
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
    },

    # Stage 4: not optimized against CHOL-EOS bilayers
    key("tail2", "cterm"): {
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
    },
    key("tail2", "chead"): {
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
    },
    key("tail2", "ctail"): {
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
    },
    key("tail2", "chme"): {
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
    },
    key("tail2", "ring"): {
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
    },
}


def pair_filename(t1: str, t2: str) -> str:
    a, b = sorted((t1, t2))
    return f"{a}-{b}.csv"


def get_stage_pairs_before(current_stage: int):
    out = set()
    for stage_num in range(1, current_stage):
        out |= STAGE_ORDER[stage_num]
    return out


def get_stage_pairs_after(current_stage: int):
    out = set()
    for stage_num in range(current_stage + 1, 5):
        out |= STAGE_ORDER[stage_num]
    return out


def get_initial_potential_source(
    t1: str,
    t2: str,
    previous_stage_pairs: set,
    out_dir: Path,
    pretrained_dir: Path,
    init_dir: Path,
):
    """
    Priority:
    1. Current FFA workflow previous-stage learned output from OUT_DIR
    2. Previously learned frozen potential from PRETRAINED_DIR
    3. Initial guess from INIT_DIR
    """
    p = key(t1, t2)

    if p in previous_stage_pairs:
        prev_csv = out_dir / pair_filename(t1, t2)
        if prev_csv.exists():
            return prev_csv, "current_stage_previous_output"

    pretrained_csv = pretrained_dir / pair_filename(t1, t2)
    if pretrained_csv.exists():
        return pretrained_csv, "pretrained_output"
    ffa_csv = FFA_PRETRAINED_DIR / pair_filename(t1, t2)
    if ffa_csv.exists():
        return ffa_csv, "ffa_pretrained_output"

    init_file = find_table_file(init_dir, t1, t2)
    return init_file, "init_dir"

def patch_main_branch_pair_exclude_states():
    """
    Add experimental-style exclude-states support to main-branch msibi Pair.

    This reproduces:
      - pair.set_state_params(state=..., optimize_against=True/False)

    It does NOT reproduce bonded exclusion features like:
      - exclude_bond_depth
      - exclude_all_bonded
    """

    PairCls = msibi.forces.Pair
    original_add_state = PairCls._add_state
    original_compute_current_distribution = PairCls._compute_current_distribution
    original_save_current_distribution = PairCls._save_current_distribution

    def patched_save_current_distribution(self, state, iteration):
        # Skip states that were intentionally excluded for this pair
        if state not in self._states:
            return
        return original_save_current_distribution(self, state, iteration)
    
    def patched_compute_current_distribution(self, state):
        # Skip states that were intentionally excluded for this pair
        if state not in self._states:
            return
        return original_compute_current_distribution(self, state)

    def set_state_params(
        self,
        state,
        optimize_against=True,
        exclude_bond_depth=None,
        exclude_all_bonded=False,
    ):
        if not hasattr(self, "_pending_state_params"):
            self._pending_state_params = {}

        self._pending_state_params[state] = {
            "optimize_against": optimize_against,
        }

        if hasattr(self, "_states") and state in self._states:
            self._states[state].update(self._pending_state_params[state])

    def patched_add_state(self, state):
        # Read pending config before doing anything
        optimize_against = True
        if hasattr(self, "_pending_state_params") and state in self._pending_state_params:
            optimize_against = self._pending_state_params[state].get("optimize_against", True)

        # If this state is excluded for this pair, do not attach it at all
        if not optimize_against:
            return

        # Otherwise do normal main-branch behavior
        original_add_state(self, state)

        # Keep the flag stored for completeness
        if "optimize_against" not in self._states[state]:
            self._states[state]["optimize_against"] = True

        if hasattr(self, "_pending_state_params") and state in self._pending_state_params:
            self._states[state].update(self._pending_state_params[state])

    def patched_update_potential(self):
        state_count = 0

        for state in self._states:
            if not self._states[state].get("optimize_against", True):
                continue

            current_dist = self._states[state]["current_distribution"]
            target_dist = self._states[state]["target_distribution"]
            self._states[state]["distribution_history"].append(current_dist)

            alpha_array = state.alpha(pot_x_range=self.x_range, dx=self.dx)
            self._potential += alpha_array * (
                state.kT * np.log(current_dist[:, 1] / target_dist[:, 1])
            )
            state_count += 1

        if state_count == 0:
            raise RuntimeError(
                f"No active optimize_against states for pair {self.name}."
            )

        self._potential /= state_count

        if isinstance(self, msibi.forces.Pair):
            self._potential, head_cut, tail_cut, real_indices = pair_corrections(
                x=self.x_range,
                V=self.potential,
                r_switch=self.r_switch,
                smoothing_window=self.smoothing_window,
                smoothing_order=self.smoothing_order,
                fit_window_size=self.correction_fit_window,
                maxfev=self.maxfev,
                head_correction_func=self.correction_form,
            )
        else:
            self._potential, head_cut, tail_cut, real_indices = bonded_corrections(
                x=self.x_range,
                V=self.potential,
                smoothing_window=self.smoothing_window,
                smoothing_order=self.smoothing_order,
                fit_window_size=self.correction_fit_window,
                maxfev=self.maxfev,
                head_correction_func=self.correction_form,
                tail_correction_func=self.correction_form,
            )

        self.potential_history.append(np.copy(self.potential))
        self._head_correction_history.append(np.copy(self.potential[0:head_cut]))
        self._tail_correction_history.append(np.copy(self.potential[tail_cut:]))
        self._learned_potential_history.append(np.copy(self.potential[real_indices]))

    PairCls.set_state_params = set_state_params
    PairCls._add_state = patched_add_state
    PairCls._compute_current_distribution = patched_compute_current_distribution
    PairCls._save_current_distribution = patched_save_current_distribution
    PairCls._update_potential = patched_update_potential


def patch_main_branch_msibi_recompute_distribution():
    MSIBICls = MSIBI

    def patched_recompute_distribution(self, force):
        for state in self.states:
            if state not in force._states:
                continue

            force._compute_current_distribution(state)
            force._save_current_distribution(state, iteration=self.n_iterations)

            print(
                f"Force: {force.name}, State: {state.name}, Iteration: {self.n_iterations}, "
                f"Fit score:{force._states[state]['f_fit'][self.n_iterations]:.6f}\n"
            )

    MSIBICls._recompute_distribution = patched_recompute_distribution

def remove_states_dir():
    states_dir = Path("states")
    if states_dir.exists():
        print("Removing existing states/ directory...")
        shutil.rmtree(states_dir)

def main(current_stage):

    patch_main_branch_pair_exclude_states()
    patch_main_branch_msibi_recompute_distribution()

    remove_states_dir()

    print(f"RUNNING CHOL STAGE {current_stage}")

    optimizer = MSIBI(
        nlist=hoomd.md.nlist.Cell,
        integrator_method=hoomd.md.methods.ConstantVolume,
        thermostat=hoomd.md.methods.thermostats.MTTK,
        thermostat_kwargs={"tau": 0.1},
        method_kwargs={},
        dt=0.0001,
        gsd_period=4000,
    )

    # ===== States (UNCHANGED initialization) =====
    stateA = State(
        name="bulk_EOS",
        kT=9.9,
        traj_file="/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states/bulk_EOS/cg-traj-wrapped.gsd",
        n_frames=300,
        sampling_stride=3,
        alpha0=0.14,
        exclude_bonded=True,
    )

    stateB = State(
        name="bulk_EOS_CHOL",
        kT=9.9,
        traj_file="/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states_copy/bulk_EOS_CHOL/cg-traj-wrapped-ringcollapsed.gsd",
        n_frames=300,
        sampling_stride=3,
        alpha0=0.14,
        exclude_bonded=True,
    )

    stateC = State(
        name="bulk_EOS_FFA",
        kT=9.9,
        traj_file="/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states/bulk_EOS_FFA/cg-traj-wrapped.gsd",
        n_frames=300,
        sampling_stride=3,
        alpha0=0.14,
        exclude_bonded=True,
    )

    stateD = State(
        name="EOS_CHOL_1_1_bilayer",
        kT=6.06,
        traj_file="/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states_copy/EOS_CHOL_BILAYER_1_1/cg-traj-wrapped-ringcollapsed.gsd",
        n_frames=300,
        sampling_stride=3,
        alpha0=0.42,
        exclude_bonded=True,
    )

    stateE = State(
        name="EOS_CHOL_1_2_bilayer",
        kT=6.06,
        traj_file="/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states_copy/EOS_CHOL_BILAYER_1_2/cg-traj-wrapped-ringcollapsed.gsd",
        n_frames=300,
        sampling_stride=3,
        alpha0=0.42,
        exclude_bonded=True,
    )

    stateF = State(
        name="EOS_pure_extended",
        kT=6.06,
        traj_file="/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states/EOS_EXTENDED_BILAYER/cg-traj-wrapped.gsd",
        n_frames=300,
        sampling_stride=3,
        alpha0=0.42,
        exclude_bonded=True,
    )

    stateG = State(
        name="EOS_LPP",
        kT=6.06,
        traj_file="/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states_copy/EOS_CE_LPP/cg-traj-wrapped-ringcollapsed.gsd",
        n_frames=300,
        sampling_stride=3,
        alpha0=0.71,
        exclude_bonded=True,
    )

    # Choose which states to run by commenting/uncommenting:
    #optimizer.add_state(stateA)
    optimizer.add_state(stateB)
    #optimizer.add_state(stateC)
    optimizer.add_state(stateD)
    optimizer.add_state(stateE)
    #optimizer.add_state(stateF)
    optimizer.add_state(stateG)

    state_lookup = {
        s.name: s
        for s in optimizer.states
    }
    PAIR_IGNORE_STATES = expand_pair_ignore_states(RAW_PAIR_IGNORE_STATES)

    bond_angle_source_gsd = "/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states/EOS_CE_LPP/cg-traj-wrapped.gsd"

    bond_json = "/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states/bonds.json"
    angle_json = "/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states/angles.json"

    with open(bond_json) as f:
        bond_forces = json.load(f)

    with open(angle_json) as f:
        angle_forces = json.load(f)

    bond_types, angle_types = get_bond_angle_types(bond_angle_source_gsd)

    bond_dict = build_bond_dict(bond_types, bond_forces)
    angle_dict = build_angle_dict(angle_types, angle_forces)

    add_bonds_from_dict(optimizer, bond_dict)
    add_angles_from_dict(optimizer, angle_dict)

    print("Loaded bond types:", sorted(bond_dict.keys()))
    print("Loaded angle types:", sorted(angle_dict.keys()))
   
   #  Adding fixed potentials (PAIR_DIR)
    stage_controlled_pairs = set()
    for a, b in ALL_STAGE_PAIRS:
        for x, y in expand_ring_pairs(a, b):
            stage_controlled_pairs.add(key(x, y))    
    pairs_added = 0
    added_keys = set()

    for source_dir in [PRETRAINED_DIR, FFA_PRETRAINED_DIR]:
        for file in source_dir.glob("*.csv"):

            if file.name.startswith("step"):
                continue
            if "-" not in file.stem:
                continue

            type1, type2 = parse_types(file.name)

            if type1 in RINGS or type2 in RINGS:
                print(f"Skipping old ring-specific pretrained pair {type1}-{type2}")
                continue

            p = key(type1, type2)

            if p in added_keys:
                continue

            if p in stage_controlled_pairs:
                print(f"Skipping {type1}-{type2} (handled by staged section)")
                continue

            r_cut, nbins = infer_rcut_nbins(file)

            pair = Pair(
                type1=type1,
                type2=type2,
                optimize=False,
                r_cut=r_cut,
                nbins=nbins
            )

            pair.set_from_file(str(file))

            optimizer.add_force(pair)
            added_keys.add(p)
            pairs_added += 1

            print(f"Loaded FIXED pretrained pair {type1}-{type2} from {file.name}")
    #  Adding extra fixed files 
    added_keys = set()
    for f in getattr(optimizer, "forces", []):
        if isinstance(f, msibi.forces.Pair):
            added_keys.add(key(f.type1, f.type2))

    extra_files = ["sticky-water.txt", "ester-water.txt", "taild-water.txt", "tail2-water.txt"]
    extra_added = 0

    for name in extra_files:
        file = EXTRA_DIR / name
        if not file.exists():
            raise FileNotFoundError(f"Missing: {file}")

        type1, type2 = parse_types(file.name)

        if key(type1, type2) in added_keys:
            print(f"Skipping duplicate {type1}-{type2} (already added)")
            continue

        pretrained_csv = PRETRAINED_DIR / pair_filename(type1, type2)

        if pretrained_csv.exists():
            r_cut, nbins = infer_rcut_nbins(pretrained_csv)
            pair = Pair(type1=type1, type2=type2, optimize=False, r_cut=r_cut, nbins=nbins)
            pair.set_from_file(str(pretrained_csv))
            print(f"Loaded FIXED pretrained EOS pair {type1}-{type2} from {pretrained_csv.name}")
        else:
            r_cut, nbins = infer_rcut_nbins(file)
            pair = Pair(type1=type1, type2=type2, optimize=False, r_cut=r_cut, nbins=nbins)
            csv_path = hoomd_table_to_msibi_csv(file, file.with_suffix(".msibi.csv"))
            pair.set_from_file(str(csv_path))
            print(f"Loaded FIXED fallback init pair {type1}-{type2} from {file.name}")

        optimizer.add_force(pair)

        added_keys.add(key(type1, type2))
        extra_added += 1

    print(f"Added fixed potentials: PAIR_DIR={pairs_added} tables, EXTRA_DIR={extra_added} tables")

    for file in PRETRAINED_DIR.glob("*.csv"):
        type1, type2 = parse_types(file.name)
        if type1 in RINGS or type2 in RINGS:
            print(f"Skipping old ring-specific pretrained pair {type1}-{type2}")
            continue
        p = key(type1, type2)
        if p in added_keys:
            continue

        if p in stage_controlled_pairs:
            continue 
        r_cut, nbins = infer_rcut_nbins(file)

        pair = Pair(type1=type1,type2=type2,optimize=False,r_cut=r_cut,nbins=nbins)
        pair.set_from_file(str(file))

        optimizer.add_force(pair)
        added_keys.add(p)

        print(f"Loaded GLOBAL PRETRAINED pair {type1}-{type2}")

    # This adds any pair present in the active states that was not already loaded
    # from PAIR_DIR, EXTRA_DIR, or PRETRAINED_DIR.

    state_type_sets_tmp = [gsd_types(s.traj_file) for s in optimizer.states]
    types_in_any_state_tmp = set().union(*state_type_sets_tmp) if state_type_sets_tmp else set()
    for t1 in sorted(types_in_any_state_tmp):
        for t2 in sorted(types_in_any_state_tmp):
            p = key(t1, t2)
            if p in stage_controlled_pairs:
                continue

            if p in added_keys:
                continue
            try:
                source_file = find_table_file(INIT_DIR, t1, t2)
            except FileNotFoundError:
                print(f"WARNING: No fallback init table for missing pair {t1}-{t2}")
                continue
            r_cut, nbins = infer_rcut_nbins(source_file)
            pair = Pair(type1=t1,type2=t2,optimize=False,r_cut=r_cut,nbins=nbins,)
            csv_path = hoomd_table_to_msibi_csv(source_file,source_file.with_suffix(".msibi.csv"))
            pair.set_from_file(str(csv_path))
            optimizer.add_force(pair)
            added_keys.add(p)
            print(f"Loaded GLOBAL INIT fallback pair {t1}-{t2} from {source_file.name}")

    #  Types in all states 
    state_type_sets = [gsd_types(s.traj_file) for s in optimizer.states]
    types_in_all_states = set.intersection(*state_type_sets) if state_type_sets else set()
    print("Types in all states:", sorted(types_in_all_states), "Number of types:", len(types_in_all_states))

    #  Adding to-be-optimized pair potentials
    safe_rmax = compute_safe_rmax(optimizer.states)
    print("Global safe r_cut:", safe_rmax)

    _, types_in_any_state, types_in_all_states = compute_types_sets(optimizer.states)
    print("Types in ANY state:", sorted(types_in_any_state))
    print("Types in ALL states:", sorted(types_in_all_states))

    OUT_DIR = Path("msibi_outputs")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    current_stage_pairs = STAGE_ORDER[current_stage]
    previous_stage_pairs = get_stage_pairs_before(current_stage)
    later_stage_pairs = get_stage_pairs_after(current_stage)

    # Expand all stage pairs into concrete ring1..ring4 pairs
    expanded_all = set()
    for a, b in ALL_STAGE_PAIRS:
        for x, y in expand_ring_pairs(a, b):
            expanded_all.add(key(x, y))
    all_candidate_pairs = expanded_all

    # Expand current-stage pairs
    expanded_current = set()
    for a, b in current_stage_pairs:
        for x, y in expand_ring_pairs(a, b):
            expanded_current.add(key(x, y))
    current_stage_pairs = expanded_current

    # Expand previous-stage pairs
    expanded_previous = set()
    for a, b in previous_stage_pairs:
        for x, y in expand_ring_pairs(a, b):
            expanded_previous.add(key(x, y))
    previous_stage_pairs = expanded_previous

    # Expand later-stage pairs
    expanded_later = set()
    for a, b in later_stage_pairs:
        for x, y in expand_ring_pairs(a, b):
            expanded_later.add(key(x, y))
    later_stage_pairs = expanded_later

    # Refresh added_keys
    added_keys = set()
    for f in getattr(optimizer, "forces", []):
        if isinstance(f, msibi.forces.Pair):
            added_keys.add(key(f.type1, f.type2))

    opt_added = 0
    fixed_added = 0
    skipped_impossible = 0

    for t1, t2 in sorted(all_candidate_pairs):
        p = key(t1, t2)

        if p in added_keys:
            print(f"Skipping {t1}-{t2} (already added earlier)")
            continue

        if (t1 not in types_in_any_state) or (t2 not in types_in_any_state):
            print(f"Skipping {t1}-{t2} (not present in ANY state)")
            skipped_impossible += 1
            continue

        present_in_all = (t1 in types_in_all_states) and (t2 in types_in_all_states)
        in_current_stage = p in current_stage_pairs
        in_previous_stage = p in previous_stage_pairs
        in_later_stage = p in later_stage_pairs

        do_optimize = (
            in_current_stage
            and present_in_all
            and (p not in MANUAL_FORCE_FIXED)
        )

        source_file, source_kind = get_initial_potential_source(
            t1=t1,
            t2=t2,
            previous_stage_pairs=previous_stage_pairs,
            out_dir=OUT_DIR,
            pretrained_dir=PRETRAINED_DIR,
            init_dir=INIT_DIR,
        )

        r_cut, nbins = infer_rcut_nbins(source_file)
        r_cut = min(r_cut, safe_rmax)

        ignore_states = [
            state_lookup[name]
            for name in PAIR_IGNORE_STATES.get(p, set())
            if name in state_lookup
        ]
        used_states = [s for s in optimizer.states if s not in ignore_states]

        used_state_names = sorted([s.name for s in used_states])
        ignored_state_names = sorted([s.name for s in ignore_states])

        status = "OPTIMIZE" if do_optimize else "FIXED"

        print(f"[{status}] {t1}-{t2}")
        print(f"source: {source_kind} -> {source_file.name}")
        print(f"used states: {used_state_names}")
        print(f"ignored states: {ignored_state_names}")

        pair = Pair(
            type1=t1,
            type2=t2,
            optimize=do_optimize,
            r_cut=r_cut,
            nbins=nbins,
            smoothing_window=15,
            correction_fit_window=7,
            smoothing_order=2,
            r_switch=r_cut - 0.1,
            head_correction_form=linear,
        )

        for s in optimizer.states:
            if s in ignore_states:
                pair.set_state_params(
                    state=s,
                    optimize_against=False,
                )
            else:
                pair.set_state_params(
                    state=s,
                    optimize_against=True,
                )

        if source_file.suffix == ".csv":
            pair.set_from_file(str(source_file))
        else:
            csv_path = hoomd_table_to_msibi_csv(
                source_file,
                source_file.with_suffix(".msibi.csv")
            )
            pair.set_from_file(str(csv_path))

        optimizer.add_force(pair)
        added_keys.add(p)

        if do_optimize:
            opt_added += 1
            print(f"Added OPTIMIZE {t1}-{t2} (source={source_kind}, file={source_file.name})")
        else:
            fixed_added += 1
            reasons = []
            if in_previous_stage:
                reasons.append("previous stage")
            if in_later_stage:
                reasons.append("later stage")
            if not present_in_all:
                reasons.append("not present in all states")
            if p in MANUAL_FORCE_FIXED:
                reasons.append("manually forced fixed")
            print(f"Added FIXED {t1}-{t2} ({', '.join(reasons)}; source={source_kind}, file={source_file.name})")

    print(f"Added optimize=True pairs: {opt_added}")
    print(f"Added optimize=False (fixed) pairs: {fixed_added}")
    print(f"Skipped (type missing in any state): {skipped_impossible}")

    OUT_DIR = Path("msibi_outputs")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    step_schedule = [100000, 100000, 250000, 250000, 500000, 500000, 1000000, 1000000, 1000000, 1000000]
    n_outer_loops = 10
    iterations_per_loop = 1

    try:
        for i in range(n_outer_loops):
            n_steps = step_schedule[i]
            print(f"Staring cycle {i+1}/{n_outer_loops} with steps = {n_steps}")
            optimizer.run_optimization(n_iterations=iterations_per_loop, n_steps=n_steps, backup_trajectories=False)
            for f in optimizer.pairs:
                if f.optimize:
                    print(f"smoothing potential: {f.name}")
                    f.smooth_potential()
    # 100K steps 
    except Exception as e:
        errfile = OUT_DIR / "failure_traceback.txt"
        with open(errfile, "w") as fh:
            fh.write(f"{type(e).__name__}: {e}\n\n")
            fh.write(traceback.format_exc())

        print(f"optimisation failed: {type(e).__name__}: {e}")
        traceback.print_exc()

        for f in optimizer.pairs:
            if f.optimize:
                tag = f.name
                f.save_potential(str(OUT_DIR / f"{tag}_failed.csv"))
                f.save_potential_history(str(OUT_DIR / f"{tag}_failed.npy"))
                for s in f._states:
                    out_npz = OUT_DIR / f"state_{s.name}__pair_{tag}_failed.npz"
                    f.save_state_data(s, out_npz)
    for f in optimizer.pairs:
        if f.optimize:
            tag = f.name
            f.save_potential(str(OUT_DIR / f"{tag}.csv"))
            f.save_potential_history(str(OUT_DIR / f"{tag}_history.npy"))
            for s in f._states:
                out_npz = OUT_DIR / f"state_{s.name}__pair_{tag}.npz"
                f.save_state_data(s, out_npz)

if __name__ == "__main__":
    main(1)
    main(2)
    main(3)
    main(4)
