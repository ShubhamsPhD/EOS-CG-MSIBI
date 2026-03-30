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

warnings.filterwarnings("ignore")

PAIR_DIR = Path("/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/cg")
EXTRA_DIR = Path("/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/eosFF_6/0")
INIT_DIR = Path("/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/eosFF_6/0")

RINGS = ("ring1", "ring2", "ring3", "ring4")


def deg(x: float) -> float:
    return np.deg2rad(x)


def key(t1: str, t2: str):
    return tuple(sorted((t1, t2)))


def parse_types(filename: str):
    stem = Path(filename).stem
    type1, type2 = stem.split("-")
    return type1, type2


def infer_rcut_nbins(filepath: Path):
    # Supports tables with comments too.
    data = np.loadtxt(filepath, comments=["#", "@"])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    r = data[:, 0]
    return float(r.max()), int(len(r))


def hoomd_table_to_msibi_csv(txt_path: Path, out_csv: Path) -> Path:
    """
    Convert HOOMD-style table (r, U [, F]) into MSIBI CSV format.

    NOTE: You previously said "scaling removed for now", so this keeps:
        x = r
        potential = U
    but keeps force scaling (your existing behavior).
    """
    D = 6.0
    E = 0.1

    arr = np.loadtxt(txt_path, comments=["#", "@"])
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise ValueError(f"{txt_path} must have at least 2 columns: r U")

    r = arr[:, 0]
    U = arr[:, 1]

    # Scaling removed
    x_reduced = r
    U_reduced = U

    df = pd.DataFrame({"x": x_reduced, "potential": U_reduced})

    if arr.shape[1] >= 3:
        F = arr[:, 2]
        F_reduced = F * D / E
        df["force"] = F_reduced

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
    """
    Expand pairs containing 'ring' into ring1..ring4.
    - ('ring', X) -> [(ring1,X), (ring2,X), (ring3,X), (ring4,X)]
    - ('ring','ring') -> all unique combos: (ring1,ring1), (ring1,ring2), ... (ring4,ring4)
    - otherwise -> [(t1,t2)]
    """
    if t1 == "ring" and t2 == "ring":
        out = []
        for i, r1 in enumerate(RINGS):
            for r2 in RINGS[i:]:
                out.append((r1, r2))
        return out

    if t1 == "ring":
        return [(r, t2) for r in RINGS]
    if t2 == "ring":
        return [(t1, r) for r in RINGS]

    return [(t1, t2)]

STAGE_1 = {
    key("cterm", "sticky"),
    key("sticky", "sticky"),
    key("oh1", "sticky"),
    key("mhead2", "sticky"),
    key("chme", "sticky"),
    key("oh2", "sticky"),
    key("ring", "sticky"),
    key("ter2", "sticky"),
    key("ester", "sticky"),
    key("amide", "sticky"),
    key("taild", "sticky"),
    key("tail", "sticky"),
    key("chead", "sticky"),
    key("tail2", "sticky"),
    key("ctail", "sticky"),
    key("head", "sticky"),
}

STAGE_2 = {
    key("ester", "taild"),
    key("ester", "tail2"),
    key("ester", "ester"),
    key("oh1", "ester"),
    key("ester", "chead"),
    key("ester", "ctail"),
    key("mhead2", "ester"),
    key("ester", "sticky"),
    key("ester", "chme"),
    key("ester", "cterm"),
    key("ester", "head"),
    key("amide", "ester"),
    key("oh2", "ester"),
    key("ester", "tail"),
    key("ester", "ring"),
    key("ester", "ter2"),
}

STAGE_3 = {
    key("ester", "taild"),
    key("taild", "ring"),
    key("oh1", "taild"),
    key("taild", "head"),
    key("taild", "cterm"),
    key("taild", "chme"),
    key("tail", "taild"),
    key("tail2", "taild"),
    key("taild", "ctail"),
    key("mhead2", "taild"),
    key("ter2", "taild"),
    key("taild", "chead"),
    key("oh2", "taild"),
    key("taild", "sticky"),
    key("amide", "taild"),
    key("taild", "taild"),
}

STAGE_4 = {
    key("ester", "tail2"),
    key("oh1", "tail2"),
    key("tail2", "cterm"),
    key("tail2", "head"),
    key("tail2", "taild"),
    key("tail2", "tail2"),
    key("tail2", "tail"),
    key("mhead2", "tail2"),
    key("tail2", "ring"),
    key("tail2", "ter2"),
    key("tail2", "chead"),
    key("oh2", "tail2"),
    key("tail2", "ctail"),
    key("amide", "tail2"),
    key("tail2", "sticky"),
    key("tail2", "chme"),
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
    "bulk_EOS",
    "bulk_EOS_CHOL",
    "bulk_EOS_FFA",
    "EOS_CHOL_1_1_bilayer",
    "EOS_CHOL_1_2_bilayer",
    "EOS_pure_extended",
    "EOS_LPP",
}

# Pairs ignores states they arent properly defined in.
# In the dict below, we ignore ALL States besides the ones inside the { }.

RAW_PAIR_IGNORE_STATES = {

    # STAGE 1

    key("cterm", "sticky"): ALL_STATES - {
        "bulk_EOS_CHOL",
        "EOS_LPP",
    },

    key("sticky", "sticky"): set(),

    key("oh1", "sticky"): ALL_STATES - {
        "bulk_EOS",
        "bulk_EOS_CHOL",
        "bulk_EOS_FFA",
        "EOS_pure_extended",
        "EOS_LPP",
    },

    key("mhead2", "sticky"): ALL_STATES - {
        "bulk_EOS",
        "bulk_EOS_CHOL",
        "bulk_EOS_FFA",
        "EOS_pure_extended",
        "EOS_LPP",
    },

    key("chme", "sticky"): ALL_STATES - {
        "bulk_EOS_CHOL",
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
        "EOS_LPP",
    },

    key("oh2", "sticky"): ALL_STATES - {
        "bulk_EOS",
        "bulk_EOS_CHOL",
        "bulk_EOS_FFA",
        "EOS_pure_extended",
        "EOS_LPP",
    },

    key("ring", "sticky"): ALL_STATES - {
        "bulk_EOS_CHOL",
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
        "EOS_LPP",
    },

    key("ter2", "sticky"): ALL_STATES - {
        "bulk_EOS",
        "bulk_EOS_CHOL",
        "bulk_EOS_FFA",
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
        "EOS_LPP",
    },

    key("ester", "sticky"): set(),

    key("amide", "sticky"): ALL_STATES - {
        "bulk_EOS",
        "bulk_EOS_CHOL",
        "bulk_EOS_FFA",
        "EOS_pure_extended",
        "EOS_LPP",
    },

    key("taild", "sticky"): ALL_STATES - {
        "bulk_EOS",
        "bulk_EOS_CHOL",
        "bulk_EOS_FFA",
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
        "EOS_LPP",
    },

    key("tail", "sticky"): set(),

    key("chead", "sticky"): ALL_STATES - {
        "bulk_EOS_CHOL",
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
        "EOS_LPP",
    },

    key("tail2", "sticky"): set(),

    key("ctail", "sticky"): ALL_STATES - {
        "bulk_EOS_CHOL",
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
        "EOS_LPP",
    },

    key("head", "sticky"): ALL_STATES - {
        "bulk_EOS_FFA",
        "EOS_LPP",
    },
    
    # STAGE 2

    key("ester", "taild"): ALL_STATES - {
        "bulk_EOS",
        "bulk_EOS_CHOL",
        "bulk_EOS_FFA",
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
        "EOS_LPP",
    },

    key("ester", "tail2"): set(),

    key("ester", "ester"): set(),

    key("oh1", "ester"): ALL_STATES - {
        "bulk_EOS",
        "bulk_EOS_CHOL",
        "bulk_EOS_FFA",
        "EOS_pure_extended",
        "EOS_LPP",
    },

    key("ester", "chead"): ALL_STATES - {
        "bulk_EOS_CHOL",
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
        "EOS_LPP",
    },

    key("ester", "ctail"): ALL_STATES - {
        "bulk_EOS_CHOL",
        "EOS_CHOL_1_2_bilayer",
        "EOS_LPP",
    },

    key("mhead2", "ester"): ALL_STATES - {
        "bulk_EOS",
        "bulk_EOS_CHOL",
        "bulk_EOS_FFA",
        "EOS_pure_extended",
        "EOS_LPP",
    },

    key("ester", "sticky"): set(),

    key("ester", "chme"): ALL_STATES - {
        "bulk_EOS_CHOL",
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
        "EOS_LPP",
    },

    key("ester", "cterm"): ALL_STATES - {
        "bulk_EOS_CHOL",
        "EOS_CHOL_1_1_bilayer",
        "EOS_LPP",
    },

    key("ester", "head"): ALL_STATES - {
        "bulk_EOS_FFA",
        "EOS_LPP",
    },

    key("amide", "ester"): ALL_STATES - {
        "bulk_EOS",
        "bulk_EOS_CHOL",
        "bulk_EOS_FFA",
        "EOS_pure_extended",
        "EOS_LPP",
    },

    key("oh2", "ester"): ALL_STATES - {
        "bulk_EOS",
        "bulk_EOS_CHOL",
        "bulk_EOS_FFA",
        "EOS_pure_extended",
        "EOS_LPP",
    },

    key("ester", "tail"): set(),

    key("ester", "ring"): ALL_STATES - {
        "bulk_EOS_CHOL",
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
        "EOS_LPP",
    },

    key("ester", "ter2"): ALL_STATES - {
        "bulk_EOS",
        "bulk_EOS_CHOL",
        "bulk_EOS_FFA",
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
        "EOS_LPP",
    },

    # STAGE 3


    key("taild", "ring"): ALL_STATES - {
        "EOS_LPP",
    },

    key("oh1", "taild"): ALL_STATES - {
        "EOS_LPP",
    },

    key("taild", "head"): ALL_STATES - {
        "EOS_LPP",
    },

    key("taild", "cterm"): ALL_STATES - {
        "EOS_LPP",
    },

    key("taild", "chme"): ALL_STATES - {
        "EOS_LPP",
    },

    key("tail", "taild"): ALL_STATES - {
        "EOS_LPP",
    },

    key("tail2", "taild"): ALL_STATES - {
        "EOS_LPP",
    },

    key("taild", "ctail"): ALL_STATES - {
        "EOS_LPP",
    },

    key("mhead2", "taild"): ALL_STATES - {
        "EOS_LPP",
    },

    key("ter2", "taild"): ALL_STATES - {
        "EOS_LPP",
    },

    key("taild", "chead"): ALL_STATES - {
        "EOS_LPP",
    },

    key("oh2", "taild"): ALL_STATES - {
        "EOS_LPP",
    },

    key("taild", "sticky"): ALL_STATES - {
        "bulk_EOS",
        "bulk_EOS_CHOL",
        "bulk_EOS_FFA",
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
        "EOS_LPP",
    },

    key("amide", "taild"): ALL_STATES - {
        "EOS_LPP",
    },

    key("taild", "taild"): ALL_STATES - {
        "EOS_LPP",
    },

    key("ester", "taild"): ALL_STATES - {
        "bulk_EOS",
        "bulk_EOS_CHOL",
        "bulk_EOS_FFA",
        "EOS_CHOL_1_1_bilayer",
        "EOS_CHOL_1_2_bilayer",
        "EOS_LPP",
    },

    # STAGE 4

    key("oh1", "tail2"): ALL_STATES - {
        "EOS_LPP",
    },

    key("tail2", "cterm"): ALL_STATES - {
        "EOS_LPP",
    },

    key("tail2", "head"): ALL_STATES - {
        "EOS_LPP",
    },

    key("tail2", "taild"): ALL_STATES - {
        "EOS_LPP",
    },

    key("tail2", "tail2"): ALL_STATES - {
        "EOS_LPP",
    },

    key("tail2", "tail"): ALL_STATES - {
        "EOS_LPP",
    },

    key("mhead2", "tail2"): ALL_STATES - {
        "EOS_LPP",
    },

    key("tail2", "ring"): ALL_STATES - {
        "EOS_LPP",
    },

    key("tail2", "ter2"): ALL_STATES - {
        "EOS_LPP",
    },

    key("tail2", "chead"): ALL_STATES - {
        "EOS_LPP",
    },

    key("oh2", "tail2"): ALL_STATES - {
        "EOS_LPP",
    },

    key("tail2", "ctail"): ALL_STATES - {
        "EOS_LPP",
    },

    key("amide", "tail2"): ALL_STATES - {
        "EOS_LPP",
    },

    key("tail2", "sticky"): set(),

    key("tail2", "chme"): ALL_STATES - {
        "EOS_LPP",
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
    init_dir: Path,
):
    """
    Strict behavior:
    - previous-stage pairs must load from OUT_DIR/<pair>.csv
    - all other stage pairs load from INIT_DIR/*.txt
    """
    p = key(t1, t2)

    if p in previous_stage_pairs:
        prev_csv = out_dir / pair_filename(t1, t2)
        if not prev_csv.exists():
            raise FileNotFoundError(
                f"Missing previous-stage output for {t1}-{t2}: {prev_csv}"
            )
        return prev_csv, "previous_stage_output"

    return find_table_file(init_dir, t1, t2), "init_dir"


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

def main():

    patch_main_branch_pair_exclude_states()
    
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
        traj_file="/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states/bulk_EOS_CHOL/cg-traj-wrapped.gsd",
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
        kT=9.9,
        traj_file="/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states/EOS_CHOL_BILAYER_1_1/cg-traj-wrapped.gsd",
        n_frames=300,
        sampling_stride=3,
        alpha0=0.42,
        exclude_bonded=True,
    )

    stateE = State(
        name="EOS_CHOL_1_2_bilayer",
        kT=9.9,
        traj_file="/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states/EOS_CHOL_BILAYER_1_2/cg-traj-wrapped.gsd",
        n_frames=300,
        sampling_stride=3,
        alpha0=0.42,
        exclude_bonded=True,
    )

    stateF = State(
        name="EOS_pure_extended",
        kT=9.9,
        traj_file="/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states/EOS_EXTENDED_BILAYER/cg-traj-wrapped.gsd",
        n_frames=300,
        sampling_stride=3,
        alpha0=0.42,
        exclude_bonded=True,
    )

    stateG = State(
        name="EOS_LPP",
        kT=9.9,
        traj_file="/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states/EOS_CE_LPP/cg-traj-wrapped.gsd",
        n_frames=300,
        sampling_stride=3,
        alpha0=0.71,
        exclude_bonded=True,
    )

    # Choose which states to run by commenting/uncommenting:
    optimizer.add_state(stateA)
    optimizer.add_state(stateB)
    optimizer.add_state(stateC)
    optimizer.add_state(stateD)
    optimizer.add_state(stateE)
    optimizer.add_state(stateF)
    optimizer.add_state(stateG)

    state_lookup = {
        s.name: s
        for s in optimizer.states
    }
    PAIR_IGNORE_STATES = expand_pair_ignore_states(RAW_PAIR_IGNORE_STATES)

    # BOND POTENTIALS (FIXED)
    bond_mhead2_amide = Bond(type1="mhead2", type2="amide", optimize=False)
    bond_mhead2_amide.set_harmonic(k=190, r0=2.92)
    optimizer.add_force(bond_mhead2_amide)

    bond_amide_tail = Bond(type1="amide", type2="tail", optimize=False)
    bond_amide_tail.set_harmonic(k=103, r0=3.1)
    optimizer.add_force(bond_amide_tail)

    bond_mhead2_oh1 = Bond(type1="mhead2", type2="oh1", optimize=False)
    bond_mhead2_oh1.set_harmonic(k=45.31, r0=2.33)
    optimizer.add_force(bond_mhead2_oh1)

    bond_mhead2_oh2 = Bond(type1="mhead2", type2="oh2", optimize=False)
    bond_mhead2_oh2.set_harmonic(k=151.99, r0=2.07)
    optimizer.add_force(bond_mhead2_oh2)

    bond_mhead2_tail = Bond(type1="mhead2", type2="tail", optimize=False)
    bond_mhead2_tail.set_harmonic(k=27.57, r0=3.62)
    optimizer.add_force(bond_mhead2_tail)

    bond_tail_tail = Bond(type1="tail", type2="tail", optimize=False)
    bond_tail_tail.set_harmonic(k=20.93, r0=3.92)
    optimizer.add_force(bond_tail_tail)

    bond_ter2_tail = Bond(type1="ter2", type2="tail", optimize=False)
    bond_ter2_tail.set_harmonic(k=35.58, r0=3.41)
    optimizer.add_force(bond_ter2_tail)

    # ESTER BONDS:
    bond_ester_tail = Bond(type1="ester", type2="tail", optimize=False)
    bond_ester_tail.set_harmonic(k=171, r0=3.055)
    optimizer.add_force(bond_ester_tail)

    bond_ester_tail2 = Bond(type1="ester", type2="tail2", optimize=False)
    bond_ester_tail2.set_harmonic(k=126, r0=2.656)
    optimizer.add_force(bond_ester_tail2)

    bond_taild_taild = Bond(type1="taild", type2="taild", optimize=False)
    bond_taild_taild.set_harmonic(k=21, r0=3.920)
    optimizer.add_force(bond_taild_taild)

    bond_tail_taild = Bond(type1="tail", type2="taild", optimize=False)
    bond_tail_taild.set_harmonic(k=21, r0=3.920)
    optimizer.add_force(bond_tail_taild)

    bond_tail2_tail = Bond(type1="tail2", type2="tail", optimize=False)
    bond_tail2_tail.set_harmonic(k=36, r0=3.410)
    optimizer.add_force(bond_tail2_tail)

    bond_ester_sticky = Bond(type1="ester", type2="sticky", optimize=False)
    bond_ester_sticky.set_harmonic(k=50, r0=1.64)
    optimizer.add_force(bond_ester_sticky)

    # CHOL AND FFA BONDS:
    bond_ring1_ring4 = Bond(type1="ring1", type2="ring4", optimize=False)
    bond_ring1_ring4.set_harmonic(k=152.1, r0=4.476)
    optimizer.add_force(bond_ring1_ring4)

    bond_ring3_chme = Bond(type1="ring3", type2="chme", optimize=False)
    bond_ring3_chme.set_harmonic(k=116.2, r0=2.658)
    optimizer.add_force(bond_ring3_chme)

    bond_ring1_ring3 = Bond(type1="ring1", type2="ring3", optimize=False)
    bond_ring1_ring3.set_harmonic(k=352.3, r0=3.603)
    optimizer.add_force(bond_ring1_ring3)

    bond_ring4_ctail = Bond(type1="ctail", type2="ring4", optimize=False)
    bond_ring4_ctail.set_harmonic(k=50.3, r0=3.693)
    optimizer.add_force(bond_ring4_ctail)

    bond_ring1_ring2 = Bond(type1="ring1", type2="ring2", optimize=False)
    bond_ring1_ring2.set_harmonic(k=324.1, r0=3.574)
    optimizer.add_force(bond_ring1_ring2)

    bond_head_tail = Bond(type1="head", type2="tail", optimize=False)
    bond_head_tail.set_harmonic(k=24.296, r0=3.28)
    optimizer.add_force(bond_head_tail)

    bond_tail_oh3 = Bond(type1="tail", type2="oh3", optimize=False)
    bond_tail_oh3.set_harmonic(k=93, r0=2.4)
    optimizer.add_force(bond_tail_oh3)

    bond_ctail_cterm = Bond(type1="ctail", type2="cterm", optimize=False)
    bond_ctail_cterm.set_harmonic(k=13.2, r0=4.22)
    optimizer.add_force(bond_ctail_cterm)

    bond_ring3_ring4 = Bond(type1="ring3", type2="ring4", optimize=False)
    bond_ring3_ring4.set_harmonic(k=362.6, r0=3.205)
    optimizer.add_force(bond_ring3_ring4)

    bond_ring2_chme = Bond(type1="chme", type2="ring2", optimize=False)
    bond_ring2_chme.set_harmonic(k=65.4, r0=2.371)
    optimizer.add_force(bond_ring2_chme)

    bond_oh4_tail = Bond(type1="oh4", type2="tail", optimize=False)
    bond_oh4_tail.set_harmonic(k=542, r0=2.5)
    optimizer.add_force(bond_oh4_tail)

    bond_ring2_ring3 = Bond(type1="ring2", type2="ring3", optimize=False)
    bond_ring2_ring3.set_harmonic(k=132.2, r0=4.203)
    optimizer.add_force(bond_ring2_ring3)

    bond_chead_ring2 = Bond(type1="chead", type2="ring2", optimize=False)
    bond_chead_ring2.set_harmonic(k=245.9, r0=2.583)
    optimizer.add_force(bond_chead_ring2)

    bond_chead_ring1 = Bond(type1="chead", type2="ring1", optimize=False)
    bond_chead_ring1.set_harmonic(k=125.5, r0=4.269)
    optimizer.add_force(bond_chead_ring1)

    # ANGLE POTENTIALS (PRESENT) 
    # EOS new additions:
    angle_ester_tail_tail = Angle(type1="ester", type2="tail", type3="tail", optimize=False)
    angle_ester_tail_tail.set_harmonic(k=64, t0=deg(160.2560))
    optimizer.add_force(angle_ester_tail_tail)

    angle_tail_ester_tail2 = Angle(type1="tail", type2="ester", type3="tail2", optimize=False)
    angle_tail_ester_tail2.set_harmonic(k=25, t0=deg(143.6980))
    optimizer.add_force(angle_tail_ester_tail2)

    angle_ester_tail2_tail = Angle(type1="ester", type2="tail2", type3="tail", optimize=False)
    angle_ester_tail2_tail.set_harmonic(k=7, t0=deg(156.4170))
    optimizer.add_force(angle_ester_tail2_tail)

    angle_tail_tail_tail2 = Angle(type1="tail", type2="tail", type3="tail2", optimize=False)
    angle_tail_tail_tail2.set_harmonic(k=8, t0=deg(155.0000))
    optimizer.add_force(angle_tail_tail_tail2)

    angle_taild_tail_ter2 = Angle(type1="taild", type2="tail", type3="ter2", optimize=False)
    angle_taild_tail_ter2.set_harmonic(k=9, t0=deg(148.3690))
    optimizer.add_force(angle_taild_tail_ter2)

    angle_tail_taild_taild = Angle(type1="tail", type2="taild", type3="taild", optimize=False)
    angle_tail_taild_taild.set_harmonic(k=9, t0=deg(152.4742))
    optimizer.add_force(angle_tail_taild_taild)

    angle_tail_tail_taild_a = Angle(type1="tail", type2="tail", type3="taild", optimize=False)
    angle_tail_tail_taild_a.set_harmonic(k=22, t0=deg(149.2265))
    optimizer.add_force(angle_tail_tail_taild_a)

    angle_sticky_ester_tail = Angle(type1="sticky", type2="ester", type3="tail", optimize=False)
    angle_sticky_ester_tail.set_harmonic(k=200, t0=deg(82.5))
    optimizer.add_force(angle_sticky_ester_tail)

    angle_sticky_ester_tail2 = Angle(type1="sticky", type2="ester", type3="tail2", optimize=False)
    angle_sticky_ester_tail2.set_harmonic(k=200, t0=deg(114.5))
    optimizer.add_force(angle_sticky_ester_tail2)

    # Original:
    angle_amide_mhead2_tail = Angle(type1="amide", type2="mhead2", type3="tail", optimize=False)
    angle_amide_mhead2_tail.set_harmonic(k=5.285, t0=deg(107.0))
    optimizer.add_force(angle_amide_mhead2_tail)

    angle_mhead2_amide_tail = Angle(type1="mhead2", type2="amide", type3="tail", optimize=False)
    angle_mhead2_amide_tail.set_harmonic(k=29.0, t0=deg(106.0))
    optimizer.add_force(angle_mhead2_amide_tail)

    angle_amide_mhead2_oh1 = Angle(type1="amide", type2="mhead2", type3="oh1", optimize=False)
    angle_amide_mhead2_oh1.set_harmonic(k=8.388, t0=deg(109.0))
    optimizer.add_force(angle_amide_mhead2_oh1)

    angle_amide_mhead2_oh2 = Angle(type1="amide", type2="mhead2", type3="oh2", optimize=False)
    angle_amide_mhead2_oh2.set_harmonic(k=10.759, t0=deg(105.0))
    optimizer.add_force(angle_amide_mhead2_oh2)

    angle_oh1_mhead2_tail = Angle(type1="oh1", type2="mhead2", type3="tail", optimize=False)
    angle_oh1_mhead2_tail.set_harmonic(k=5.671, t0=deg(128.0))
    optimizer.add_force(angle_oh1_mhead2_tail)

    angle_oh2_mhead2_tail = Angle(type1="oh2", type2="mhead2", type3="tail", optimize=False)
    angle_oh2_mhead2_tail.set_harmonic(k=361.765, t0=deg(69.0))
    optimizer.add_force(angle_oh2_mhead2_tail)

    angle_oh1_mhead2_oh2 = Angle(type1="oh1", type2="mhead2", type3="oh2", optimize=False)
    angle_oh1_mhead2_oh2.set_harmonic(k=12.22, t0=deg(115.0))
    optimizer.add_force(angle_oh1_mhead2_oh2)

    angle_mhead2_tail_oh3_a = Angle(type1="mhead2", type2="tail", type3="oh3", optimize=False)
    angle_mhead2_tail_oh3_a.set_harmonic(k=231, t0=deg(52.156))
    optimizer.add_force(angle_mhead2_tail_oh3_a)

    angle_mhead2_tail_oh3_b = Angle(type1="mhead2", type2="tail", type3="oh3", optimize=False)
    angle_mhead2_tail_oh3_b.set_harmonic(k=320, t0=deg(77.332))
    optimizer.add_force(angle_mhead2_tail_oh3_b)

    angle_mhead2_tail_tail = Angle(type1="mhead2", type2="tail", type3="tail", optimize=False)
    angle_mhead2_tail_tail.set_harmonic(k=6.33, t0=deg(158.0))
    optimizer.add_force(angle_mhead2_tail_tail)

    angle_amide_tail_tail = Angle(type1="amide", type2="tail", type3="tail", optimize=False)
    angle_amide_tail_tail.set_harmonic(k=6.33, t0=deg(158.0))
    optimizer.add_force(angle_amide_tail_tail)

    angle_amide_tail_oh4 = Angle(type1="amide", type2="tail", type3="oh4", optimize=False)
    angle_amide_tail_oh4.set_harmonic(k=490, t0=deg(71.0))
    optimizer.add_force(angle_amide_tail_oh4)

    angle_oh4_tail_tail = Angle(type1="oh4", type2="tail", type3="tail", optimize=False)
    angle_oh4_tail_tail.set_harmonic(k=16, t0=deg(141.0))
    optimizer.add_force(angle_oh4_tail_tail)

    angle_tail_tail_oh3 = Angle(type1="tail", type2="tail", type3="oh3", optimize=False)
    angle_tail_tail_oh3.set_harmonic(k=54, t0=deg(143.77229953217747))
    optimizer.add_force(angle_tail_tail_oh3)

    angle_head_tail_tail = Angle(type1="head", type2="tail", type3="tail", optimize=False)
    angle_head_tail_tail.set_harmonic(k=11.63, t0=deg(148.0))
    optimizer.add_force(angle_head_tail_tail)

    angle_ter2_tail_tail = Angle(type1="tail", type2="tail", type3="ter2", optimize=False)
    angle_ter2_tail_tail.set_harmonic(k=7.93, t0=deg(155.0))
    optimizer.add_force(angle_ter2_tail_tail)

    angle_tail_tail_tail = Angle(type1="tail", type2="tail", type3="tail", optimize=False)
    angle_tail_tail_tail.set_harmonic(k=56.97, t0=deg(158.0))
    optimizer.add_force(angle_tail_tail_tail)

    # RING/ CHOL:
    angle_ring1_chead_ring2 = Angle(type1="ring1", type2="chead", type3="ring2", optimize=False)
    angle_ring1_chead_ring2.set_harmonic(k=1132.5681907540516, t0=deg(56.326))
    optimizer.add_force(angle_ring1_chead_ring2)

    angle_ring1_ring2_ring3 = Angle(type1="ring1", type2="ring2", type3="ring3", optimize=False)
    angle_ring1_ring2_ring3.set_harmonic(k=2678.769981609583, t0=deg(54.019))
    optimizer.add_force(angle_ring1_ring2_ring3)

    angle_chme_ring3_ring2 = Angle(type1="chme", type2="ring3", type3="ring2", optimize=False)
    angle_chme_ring3_ring2.set_harmonic(k=77.47422986027716, t0=deg(133.183))
    optimizer.add_force(angle_chme_ring3_ring2)

    angle_ring1_ring3_ring2 = Angle(type1="ring1", type2="ring3", type3="ring2", optimize=False)
    angle_ring1_ring3_ring2.set_harmonic(k=2787.102591159971, t0=deg(53.386))
    optimizer.add_force(angle_ring1_ring3_ring2)

    angle_chme_ring3_ring4 = Angle(type1="chme", type2="ring3", type3="ring4", optimize=False)
    angle_chme_ring3_ring4.set_harmonic(k=671.6621792124029, t0=deg(53.591))
    optimizer.add_force(angle_chme_ring3_ring4)

    angle_ring1_ring4_ring3 = Angle(type1="ring1", type2="ring4", type3="ring3", optimize=False)
    angle_ring1_ring4_ring3.set_harmonic(k=2675.4871752595714, t0=deg(52.465))
    optimizer.add_force(angle_ring1_ring4_ring3)

    angle_chead_ring2_ring3 = Angle(type1="chead", type2="ring2", type3="ring3", optimize=False)
    angle_chead_ring2_ring3.set_harmonic(k=597.4707557021375, t0=deg(139.446))
    optimizer.add_force(angle_chead_ring2_ring3)

    angle_chead_ring2_ring1 = Angle(type1="chead", type2="ring2", type3="ring1", optimize=False)
    angle_chead_ring2_ring1.set_harmonic(k=1011.1043558036172, t0=deg(85.449))
    optimizer.add_force(angle_chead_ring2_ring1)

    angle_chead_ring2_chme = Angle(type1="chead", type2="ring2", type3="chme", optimize=False)
    angle_chead_ring2_chme.set_harmonic(k=75.50454605027011, t0=deg(92.779))
    optimizer.add_force(angle_chead_ring2_chme)

    angle_ctail_ring4_ring3 = Angle(type1="ctail", type2="ring4", type3="ring3", optimize=False)
    angle_ctail_ring4_ring3.set_harmonic(k=276.08401403598765, t0=deg(94.433))
    optimizer.add_force(angle_ctail_ring4_ring3)

    # NOTE: You had angle_ring2_ring1_ring4 twice. Keep only one.
    angle_ring2_ring1_ring4 = Angle(type1="ring2", type2="ring1", type3="ring4", optimize=False)
    angle_ring2_ring1_ring4.set_harmonic(k=663.1268827023723, t0=deg(115.083))
    optimizer.add_force(angle_ring2_ring1_ring4)

    angle_ring3_ring1_ring4 = Angle(type1="ring3", type2="ring1", type3="ring4", optimize=False)
    angle_ring3_ring1_ring4.set_harmonic(k=3617.652597712942, t0=deg(44.826))
    optimizer.add_force(angle_ring3_ring1_ring4)

    angle_ring2_ring3_ring4 = Angle(type1="ring2", type2="ring3", type3="ring4", optimize=False)
    angle_ring2_ring3_ring4.set_harmonic(k=446.46166360159725, t0=deg(133.371))
    optimizer.add_force(angle_ring2_ring3_ring4)

    angle_chme_ring3_ring1 = Angle(type1="chme", type2="ring3", type3="ring1", optimize=False)
    angle_chme_ring3_ring1.set_harmonic(k=354.5430858012684, t0=deg(95.501))
    optimizer.add_force(angle_chme_ring3_ring1)

    angle_chme_ring2_ring3 = Angle(type1="chme", type2="ring2", type3="ring3", optimize=False)
    angle_chme_ring2_ring3.set_harmonic(k=436.613244551562, t0=deg(60.855))
    optimizer.add_force(angle_chme_ring2_ring3)

    angle_ring3_ring2_ring4 = Angle(type1="ring3", type2="ring2", type3="ring4", optimize=False)
    angle_ring3_ring2_ring4.set_harmonic(k=3617.652597712942, t0=deg(44.826))
    optimizer.add_force(angle_ring3_ring2_ring4)

    angle_chme_ring2_ring1 = Angle(type1="chme", type2="ring2", type3="ring1", optimize=False)
    angle_chme_ring2_ring1.set_harmonic(k=643.4300446023019, t0=deg(49.707))
    optimizer.add_force(angle_chme_ring2_ring1)

    angle_cterm_ctail_ring4 = Angle(type1="cterm", type2="ctail", type3="ring4", optimize=False)
    angle_cterm_ctail_ring4.set_harmonic(k=7.550454605027012, t0=deg(125.028))
    optimizer.add_force(angle_cterm_ctail_ring4)

    angle_ring2_ring1_ring3 = Angle(type1="ring2", type2="ring1", type3="ring3", optimize=False)
    angle_ring2_ring1_ring3.set_harmonic(k=1322.970959054733, t0=deg(70.993))
    optimizer.add_force(angle_ring2_ring1_ring3)

    angle_ring1_ring3_ring4 = Angle(type1="ring1", type2="ring3", type3="ring4", optimize=False)
    angle_ring1_ring3_ring4.set_harmonic(k=1329.5365717547566, t0=deg(81.133))
    optimizer.add_force(angle_ring1_ring3_ring4)

    angle_chead_ring1_ring4 = Angle(type1="chead", type2="ring1", type3="ring4", optimize=False)
    angle_chead_ring1_ring4.set_harmonic(k=351.2602794512566, t0=deg(151.233))
    optimizer.add_force(angle_chead_ring1_ring4)

    angle_chead_ring1_ring3 = Angle(type1="chead", type2="ring1", type3="ring3", optimize=False)
    angle_chead_ring1_ring3.set_harmonic(k=180.22606861564475, t0=deg(144.251))
    optimizer.add_force(angle_chead_ring1_ring3)

    angle_chead_ring1_ring2 = Angle(type1="chead", type2="ring1", type3="ring2", optimize=False)
    angle_chead_ring1_ring2.set_harmonic(k=2941.3944896105227, t0=deg(36.668))
    optimizer.add_force(angle_chead_ring1_ring2)

    angle_ctail_ring4_ring1 = Angle(type1="ctail", type2="ring4", type3="ring1", optimize=False)
    angle_ctail_ring4_ring1.set_harmonic(k=180.22606861564475, t0=deg(144.251))
    optimizer.add_force(angle_ctail_ring4_ring1)
 
   #  Adding fixed potentials (PAIR_DIR)
    stage_controlled_pairs = set()
    for a, b in ALL_STAGE_PAIRS:
        for x, y in expand_ring_pairs(a, b):
            stage_controlled_pairs.add(key(x, y))    

    pairs_added = 0
    for file in PAIR_DIR.glob("*.txt"):
        if file.name.startswith("step"):
            continue
        if "-" not in file.stem:
            continue

        type1, type2 = parse_types(file.name)
        p = key(type1, type2)

        if p in stage_controlled_pairs:
            print(f"Skipping {type1}-{type2} from PAIR_DIR (handled by staged section)")
            continue

        r_cut, nbins = infer_rcut_nbins(file)

        pair = Pair(type1=type1, type2=type2, optimize=False, r_cut=r_cut, nbins=nbins)
        csv_path = hoomd_table_to_msibi_csv(file, file.with_suffix(".msibi.csv"))
        pair.set_from_file(str(csv_path))
        optimizer.add_force(pair)

        pairs_added += 1

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

        r_cut, nbins = infer_rcut_nbins(file)
        pair = Pair(type1=type1, type2=type2, optimize=False, r_cut=r_cut, nbins=nbins)
        csv_path = hoomd_table_to_msibi_csv(file, file.with_suffix(".msibi.csv"))
        pair.set_from_file(str(csv_path))
        optimizer.add_force(pair)

        added_keys.add(key(type1, type2))
        extra_added += 1

    print(f"Added fixed potentials: PAIR_DIR={pairs_added} tables, EXTRA_DIR={extra_added} tables")

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

    current_stage_pairs = STAGE_ORDER[CURRENT_STAGE]
    previous_stage_pairs = get_stage_pairs_before(CURRENT_STAGE)
    later_stage_pairs = get_stage_pairs_after(CURRENT_STAGE)

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
    
    n_outer_loops = 10
    iterations_per_loop = 1

    try:
        for i in range(n_outer_loops):
            print(f"Staring cycle {i+1}/{n_outer_loops}")
            optimizer.run_optimization(n_iterations=iterations_per_loop, n_steps=1000000, backup_trajectories=False)
            for f in optimizer.pairs:
                if f.optimize:
                    print(f"smoothing potential: {f.name}")
                    f.smooth_potential()

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
    main()
