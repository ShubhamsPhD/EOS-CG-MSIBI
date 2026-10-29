import glob
from os import path
import os

for fname in glob.glob("pot*"):
    pairname = fname.split(".")[-2]
    assert path.exists(f"{pairname}.txt")
    os.system(f"mv {fname} {pairname}.txt")
