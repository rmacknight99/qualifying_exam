from utils import *
import sys

sub_dirs = sys.argv[1].split()
n_cores = int(sys.argv[2])

df = gather(sub_dirs=sub_dirs, epoxide_energy=68.80, n_cores=n_cores).replace(0, "NA")
print("", flush=True)
print(df, flush=True)
try:
    best_complex(df)
except:
    pass

df.to_csv("summary.csv", index_label="lewis_acid")
