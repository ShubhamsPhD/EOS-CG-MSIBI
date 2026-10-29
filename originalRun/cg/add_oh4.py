import shutil
import glob
import copy

files = glob.glob('*oh1*')

for x in files:
    shutil.copy2(x, x.replace('oh1', 'oh4'))
    print(x)
