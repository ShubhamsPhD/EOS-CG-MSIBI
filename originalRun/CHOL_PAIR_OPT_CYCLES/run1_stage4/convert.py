from pathlib import Path
import pandas as pd
import numpy as np

# Input folder
#input_dir = Path("/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI_GPU_RUNS/FFA_PAIR_OPT_CYCLES/run1/msibi_outputs")
#input_dir = Path("/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI_GPU_RUNS/ParallelRuns/run_1_test/ffa_stage34/run1/msibi_outputs")
input_dir = Path("/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI_GPU_RUNS/CHOL_PAIR_OPT_CYCLES/run1_stage4/msibi_outputs")
#input_dir = Path("/users/sd3008/sharedscratch/CEREOS_CG/Atomistic_targetStates/EOS_MSIBI_GPU_RUNS/CHOL_PAIR_OPT_CYCLES/run1/msibi_outputs")
# Output folder
output_dir = input_dir / "txt_tables"
output_dir.mkdir(exist_ok=True)

# Loop through all CSV files
for csv_file in input_dir.glob("*.csv"):

    # Read CSV
    df = pd.read_csv(csv_file)

    # Keep only required columns
    arr = df[["x", "potential", "force"]].to_numpy()

    # Output filename
    out_file = output_dir / f"{csv_file.stem}.txt"

    # Save in scientific notation with space separation
    np.savetxt(
        out_file,
        arr,
        fmt="%.18e"
    )

    print(f"Converted: {csv_file.name} -> {out_file.name}")

print("\nDone.")
