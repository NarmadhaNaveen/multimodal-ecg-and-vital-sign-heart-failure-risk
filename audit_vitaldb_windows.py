from pathlib import Path
import numpy as np

folder = Path("data/processed/vitaldb/case1")
files = sorted(folder.glob("window_*.npz"))

total_vital_values = 0
missing_vital_values = 0
bad_ecg_windows = 0
bad_vital_windows = 0

for f in files:
    with np.load(f, allow_pickle=False) as d:
        ecg = d["ecg"]
        vitals = d["vitals"]

        bad_ecg_windows += int(not np.isfinite(ecg).all())
        bad_vital_windows += int(not np.isfinite(vitals).all())

        total_vital_values += vitals.size
        missing_vital_values += int(np.isnan(vitals).sum())

print("Total windows:", len(files))
print("ECG windows with non-finite values:", bad_ecg_windows)
print("Windows with missing vital values:", bad_vital_windows)
print("Total vital measurements:", total_vital_values)
print("Missing vital measurements:", missing_vital_values)
print("Overall vital missing percentage:",
      round(100 * missing_vital_values / total_vital_values, 2)
      if total_vital_values else 0)
