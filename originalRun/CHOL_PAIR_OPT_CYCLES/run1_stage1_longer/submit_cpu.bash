#!/bin/bash -l
#SBATCH --job-name=ms-Chol1
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=sd3008@hw.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=7-00:00:00
#SBATCH --partition=nodes
#SBATCH --qos=eps_comsel
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.out

cd "$SLURM_SUBMIT_DIR"
echo "$SLURM_JOB_ID $SLURM_SUBMIT_DIR" >> ~/job-id-dirs.txt

flight start
flight env activate gridware
module load apps/anaconda3/2023.03/bin

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate msibi-env-test
export PYTHONPATH="$HOME/programs/hoomd-funcs:$PYTHONPATH"

python -u run_msibi_cholPairs.py
