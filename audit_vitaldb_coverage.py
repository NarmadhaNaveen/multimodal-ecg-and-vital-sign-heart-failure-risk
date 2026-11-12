from pathlib import Path
import numpy as np

folder = Path("data/processed/vitaldb/case1")
files = sorted(folder.glob("window_*.npz"))

names = ["HR", "SBP", "DBP", "MAP", "SpO2"]
valid_counts = np.zeros(len(names), dtype=np.int64)
total_counts = np.zeros(len(names), dtype=np.int64)
windows_80 = np.zeros(len(names), dtype=np.int64)
windows_all_80 = 0

for f in files:
    with np.load(f, allow_pickle=False) as d:
        v = d["vitals"]
        finite = np.isfinite(v)

        valid_counts += finite.sum(axis=0)
        total_counts += len(v)

        coverage = finite.mean(axis=0)
        windows_80 += (coverage >= 0.80)
        windows_all_80 += int(np.all(coverage >= 0.80))

print("Total windows:", len(files))
print("\nPer-channel completeness:")

for i, name in enumerate(names):
    pct = 100 * valid_counts[i] / total_counts[i]
    print(f"{name}: {pct:.1f}% valid; "
          f"{windows_80[i]} windows have >=80% valid values")

print("\nWindows with >=80% valid values in every channel:",
      windows_all_80)
