from pathlib import Path
import shutil

RINGS = ["ring1", "ring2", "ring3", "ring4"]

def canonical_name(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))

def find_table(pair_dir: Path, a: str, b: str):
    """Find a-b.txt or b-a.txt."""
    f1 = pair_dir / f"{a}-{b}.txt"
    f2 = pair_dir / f"{b}-{a}.txt"
    if f1.exists():
        return f1
    if f2.exists():
        return f2
    return None

def ensure_pair_from_canonical(pair_dir: Path, a: str, b: str, ca: str, cb: str, overwrite: bool = False) -> None:
    """Create a-b.txt from canonical ca-cb.txt."""
    src = find_table(pair_dir, ca, cb)
    if src is None:
        raise FileNotFoundError(f"Missing canonical table for {ca}-{cb} in {pair_dir}")

    dst_a, dst_b = canonical_name(a, b)
    dst = pair_dir / f"{dst_a}-{dst_b}.txt"

    if dst.exists() and not overwrite:
        return

    shutil.copyfile(src, dst)

def make_ring_duplicates(pair_dir: str, overwrite: bool = False) -> None:
    pair_dir = Path(pair_dir)

    # 1) ring-X -> ring1..ring4-X
    # Scan canonical ring-* files
    for src in pair_dir.glob("ring-*.txt"):
        stem = src.stem  # ring-ctail
        _, other = stem.split("-", 1)

        # Skip ring-ring here (handled below)
        if other == "ring":
            continue

        for r in RINGS:
            ensure_pair_from_canonical(pair_dir, r, other, "ring", other, overwrite=overwrite)

    # Also handle *-ring naming if your canonicals are X-ring.txt instead of ring-X.txt
    for src in pair_dir.glob("*-ring.txt"):
        stem = src.stem  # ctail-ring
        other, _ = stem.split("-", 1)
        if other == "ring":
            continue
        for r in RINGS:
            ensure_pair_from_canonical(pair_dir, r, other, "ring", other, overwrite=overwrite)

    # 2) ring-ring -> all ringX-ringY (10 combos)
    # Requires canonical ring-ring.txt (either ring-ring.txt only)
    rr = find_table(pair_dir, "ring", "ring")
    if rr is not None:
        for i, ra in enumerate(RINGS):
            for rb in RINGS[i:]:
                ensure_pair_from_canonical(pair_dir, ra, rb, "ring", "ring", overwrite=overwrite)

    print("Done. Created ring1..ring4 nonbonded duplicates from canonical ring tables.")

if __name__ == "__main__":
    # Example:
    # make_ring_duplicates("/Users/sd3008/Desktop/eosFF_6/0", overwrite=False)
    make_ring_duplicates(".", overwrite=False)
