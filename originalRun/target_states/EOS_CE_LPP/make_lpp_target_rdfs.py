#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import gsd.hoomd
import freud


# =========================
# USER SETTINGS
# =========================

STATE_DIR = Path(
    "/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI/target_states/EOS_CE_LPP"
)

GSD_FILE = STATE_DIR / "cg-traj-wrapped.gsd"
OUT_DIR = STATE_DIR / "target_rdfs_intermolecular"

PAIRS = [
    ("ester", "sticky"),
    ("mhead2", "sticky"),
    ("sticky", "sticky"),
    ("sticky", "tail2"),
]

BINS = 300
R_MAX = 4.0
START_FRAME = 0
STOP_FRAME = None
STRIDE = 1


# =========================
# HELPERS
# =========================

def sorted_pair_name(t1: str, t2: str) -> str:
    a, b = sorted((t1, t2))
    return f"{a}-{b}.csv"


def get_type_indices(frame, bead_type: str) -> np.ndarray:
    type_names = list(frame.particles.types)
    if bead_type not in type_names:
        return np.array([], dtype=np.int64)

    typeid = type_names.index(bead_type)
    idx = np.where(frame.particles.typeid == typeid)[0]
    return idx.astype(np.int64)


# ---------- Union-Find for connected components ----------
class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int64)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def molecule_ids_from_bonds(frame) -> np.ndarray | None:
    """
    Build molecule IDs from the bond graph in the first GSD frame.
    Returns None if no bonds are present.
    """
    n_particles = frame.particles.N

    if not hasattr(frame, "bonds"):
        return None
    if frame.bonds.N == 0:
        return None

    uf = UnionFind(n_particles)
    bond_groups = np.asarray(frame.bonds.group, dtype=np.int64)

    for i, j in bond_groups:
        uf.union(int(i), int(j))

    roots = np.array([uf.find(i) for i in range(n_particles)], dtype=np.int64)
    _, mol_ids = np.unique(roots, return_inverse=True)
    return mol_ids.astype(np.int64)


def molecule_ids_from_body(frame) -> np.ndarray | None:
    """
    Fall back to HOOMD body IDs if populated.
    Returns None if body is absent or all negative.
    """
    if not hasattr(frame.particles, "body"):
        return None

    body = np.asarray(frame.particles.body, dtype=np.int64)
    if body.size == 0:
        return None

    # In HOOMD, free particles often have body = -1
    if np.all(body < 0):
        return None

    # Make unique per-particle IDs for negative entries so they don't all collapse
    out = body.copy()
    negative = np.where(out < 0)[0]
    if len(negative) > 0:
        start = out[out >= 0].max() + 1 if np.any(out >= 0) else 0
        out[negative] = np.arange(start, start + len(negative), dtype=np.int64)

    _, mol_ids = np.unique(out, return_inverse=True)
    return mol_ids.astype(np.int64)


def get_molecule_ids(frame) -> np.ndarray:
    """
    Prefer bond-derived molecule IDs. Fall back to particles.body.
    """
    mol_ids = molecule_ids_from_bonds(frame)
    if mol_ids is not None:
        print("Molecule IDs built from bond graph in GSD.")
        return mol_ids

    mol_ids = molecule_ids_from_body(frame)
    if mol_ids is not None:
        print("Molecule IDs built from particles.body in GSD.")
        return mol_ids

    raise RuntimeError(
        "Could not determine molecule IDs from GSD. "
        "No usable bonds or body IDs were found."
    )


def build_intermol_neighbor_list(box, ref_pos, query_pos, ref_mol, query_mol, r_max, exclude_self):
    """
    Build a NeighborList using freud, then remove all same-molecule pairs.
    """
    aq = freud.locality.AABBQuery.from_system((box, ref_pos))

    query_args = {"r_max": r_max}
    if exclude_self:
        query_args["exclude_ii"] = True
    else:
        query_args["exclude_ii"] = False

    raw_nlist = aq.query(query_pos, query_args).toNeighborList()

    # Indices into ref_pos / query_pos
    point_idx = np.asarray(raw_nlist.point_indices, dtype=np.int64)
    query_idx = np.asarray(raw_nlist.query_point_indices, dtype=np.int64)

    # Keep only intermolecular pairs
    keep = ref_mol[point_idx] != query_mol[query_idx]

    if not np.any(keep):
        return aq, None

    # freud v3 NeighborList.from_arrays wants vectors, not distances
    filtered = freud.locality.NeighborList.from_arrays(
        num_query_points=len(query_pos),
        num_points=len(ref_pos),
        query_point_indices=query_idx[keep],
        point_indices=point_idx[keep],
        vectors=np.asarray(raw_nlist.vectors, dtype=np.float32)[keep],
        weights=np.ones(np.count_nonzero(keep), dtype=np.float32),
    )

    return aq, filtered

def compute_pair_rdf_intermolecular(
    gsd_file: Path,
    type1: str,
    type2: str,
    mol_ids: np.ndarray,
    bins: int = 300,
    r_max: float = 4.0,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
):
    traj = gsd.hoomd.open(str(gsd_file), mode="r")
    if len(traj) == 0:
        raise ValueError(f"No frames found in {gsd_file}")

    first = traj[0]
    box_lengths = first.configuration.box[:3]
    print(
        f"First-frame box lengths for {type1}-{type2}: "
        f"Lx={box_lengths[0]:.4f}, Ly={box_lengths[1]:.4f}, Lz={box_lengths[2]:.4f}"
    )
    print(f"Using r_max={r_max:.4f} for {type1}-{type2}")

    rdf = freud.density.RDF(bins=bins, r_max=r_max)

    n_frames = len(traj)
    if stop is None or stop > n_frames:
        stop = n_frames

    frames_used = 0
    pair_counts = []

    for i in range(start, stop, stride):
        frame = traj[i]
        positions = np.asarray(frame.particles.position, dtype=np.float32)
        box = frame.configuration.box

        idx1 = get_type_indices(frame, type1)
        idx2 = get_type_indices(frame, type2)

        if len(idx1) == 0 or len(idx2) == 0:
            continue

        pos1 = positions[idx1]
        pos2 = positions[idx2]
        mol1 = mol_ids[idx1]
        mol2 = mol_ids[idx2]

        exclude_self = (type1 == type2)
        aq, nlist = build_intermol_neighbor_list(
            box=box,
            ref_pos=pos1,
            query_pos=pos2,
            ref_mol=mol1,
            query_mol=mol2,
            r_max=r_max,
            exclude_self=exclude_self,
        )

        if nlist is None or len(nlist) == 0:
            continue

        rdf.compute(aq, neighbors=nlist, reset=False)
        frames_used += 1
        pair_counts.append(len(nlist))

    if frames_used == 0:
        raise ValueError(
            f"No usable intermolecular pairs found for {type1}-{type2} in {gsd_file}"
        )

    return (
        rdf.bin_centers.copy(),
        rdf.rdf.copy(),
        frames_used,
        float(r_max),
        float(np.mean(pair_counts)) if pair_counts else 0.0,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {GSD_FILE}")
    print(f"Writing intermolecular RDFs to: {OUT_DIR}")

    traj = gsd.hoomd.open(str(GSD_FILE), mode="r")
    first = traj[0]
    mol_ids = get_molecule_ids(first)

    print(f"Total particles: {first.particles.N}")
    print(f"Detected molecules: {len(np.unique(mol_ids))}")

    for t1, t2 in PAIRS:
        print(f"\nComputing INTERMOLECULAR RDF for {t1}-{t2}")

        r, g_r, frames_used, r_max, mean_pairs = compute_pair_rdf_intermolecular(
            gsd_file=GSD_FILE,
            type1=t1,
            type2=t2,
            mol_ids=mol_ids,
            bins=BINS,
            r_max=R_MAX,
            start=START_FRAME,
            stop=STOP_FRAME,
            stride=STRIDE,
        )

        out_file = OUT_DIR / sorted_pair_name(t1, t2)
        df = pd.DataFrame({
            "r": r,
            "rdf": g_r,
        })
        df.to_csv(out_file, index=False)

        print(f"  frames used: {frames_used}")
        print(f"  mean intermolecular neighbor pairs/frame: {mean_pairs:.1f}")
        print(f"  r_max: {r_max:.6f}")
        print(f"  saved: {out_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
