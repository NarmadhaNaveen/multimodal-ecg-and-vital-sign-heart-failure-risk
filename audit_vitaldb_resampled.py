from pathlib import Path
import numpy as np

folder = Path("data/processed/vitaldb/case1_resampled")
files = sorted(folder.glob("window_*.npz"))

print("Total windows:", len(files))

if not files:
    raise SystemExit("No windows found.")

with np.load(files[0], allow_pickle=False) as d:
    print("ECG shape:", d["ecg"].shape)
    print("Vitals shape:", d["vitals"].shape)
    print("Vital names:", d["vital_names"].tolist())
    print("ECG sampling rate:", int(d["sampling_rate"]))

names = ["HR", "SBP", "DBP", "MAP", "SpO2"]
total = np.zeros(5, dtype=np.int64)
missing = np.zeros(5, dtype=np.int64)
minimum = np.full(5, np.inf)
maximum = np.full(5, -np.inf)
bad_ecg = 0

for file in files:
    with np.load(file, allow_pickle=False) as d:
        e = d["ecg"]
        v = d["vitals"]
        bad_ecg += int(not np.isfinite(e).all())

        for i in range(5):
            x = v[:, i]
            valid = np.isfinite(x)
            total[i] += len(x)
            missing[i] += int((~valid).sum())

            if valid.any():
                minimum[i] = min(minimum[i], float(x[valid].min()))
                maximum[i] = max(maximum[i], float(x[valid].max()))

print("\nVital channel audit:")
for i, name in enumerate(names):
    pct = 100 * missing[i] / total[i]
    print(
        f"{name}: missing={pct:.2f}%, "
        f"min={minimum[i]:.2f}, max={maximum[i]:.2f}"
    )

print("\nWindows with non-finite ECG:", bad_ecg)
