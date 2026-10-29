#!/usr/bin/env python3
import os
import numpy as np
import gsd.hoomd

state_dirs = [
    "bulk_EOS",
    "bulk_EOS_CHOL",
    "bulk_EOS_FFA",
    "EOS_CE_LPP",
    "EOS_CHOL_BILAYER_1_1",
    "EOS_CHOL_BILAYER_1_2",
    "EOS_EXTENDED_BILAYER",
]

for state in state_dirs:
    gsd_file = os.path.join(state, "cg-traj-wrapped.gsd")
    print(f"\n=== {state} ===")

    if not os.path.isfile(gsd_file):
        print(f"Missing: {gsd_file}")
        continue

    try:
        with gsd.hoomd.open(gsd_file, "r") as t:
            fr = t[0]
            group = np.asarray(fr.bonds.group)

            print("N bonds:", len(group))

            if len(group) == 0:
                print("No bonds found")
                continue

            first_col_sorted = np.all(group[1:, 0] >= group[:-1, 0])
            internal_sorted = np.all(group[:, 0] <= group[:, 1])

            print("First column sorted:", first_col_sorted)
            print("Internal pair sorted:", internal_sorted)

            bad_first = np.where(group[1:, 0] < group[:-1, 0])[0]
            bad_internal = np.where(group[:, 0] > group[:, 1])[0]

            print("Number of bad first-column transitions:", len(bad_first))
            print("Number of internally unsorted bonds:", len(bad_internal))

            print("First 20 bonds:")
            print(group[:20])

            if len(bad_first) > 0:
                print("Example bad first-column transitions:")
                for idx in bad_first[:10]:
                    print(f"  group[{idx}]   = {group[idx]}")
                    print(f"  group[{idx+1}] = {group[idx+1]}")

            if len(bad_internal) > 0:
                print("Example internally unsorted bonds:")
                print(group[bad_internal[:10]])

    except Exception as e:
        print(f"ERROR reading {gsd_file}: {e}")
