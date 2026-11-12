import vitaldb
import numpy as np
from pathlib import Path

vf = vitaldb.VitalFile("data/raw/vitaldb/1.vital")

out = Path("data/processed/vitaldb/case1")
out.mkdir(parents=True, exist_ok=True)

# ECG sampled at 200 Hz; vital signs sampled at 1 Hz.
ecg = np.asarray(
    vf.to_numpy(["SNUADC/ECG_II"], interval=0.005)
).reshape(-1)

tracks = {
    "HR": ("Solar8000/HR", 30, 220),
    "SBP": ("Solar8000/ART_SBP", 50, 250),
    "DBP": ("Solar8000/ART_DBP", 20, 150),
    "MAP": ("Solar8000/ART_MBP", 30, 180),
    "SpO2": ("Solar8000/PLETH_SPO2", 50, 100),
}

vitals = {}
for key, (name, low, high) in tracks.items():
    x = np.asarray(vf.to_numpy([name], interval=1.0)).reshape(-1)
    x[~np.isfinite(x) | (x < low) | (x > high)] = np.nan
    vitals[key] = x.astype(np.float32)

# Use the common duration across all tracks.
seconds = min(len(ecg) // 200, *(len(x) for x in vitals.values()))
window_seconds = 30
saved = 0

for start in range(0, seconds - window_seconds + 1, window_seconds):
    end = start + window_seconds

    e = ecg[start * 200:end * 200]
    v = np.column_stack([x[start:end] for x in vitals.values()])

    # Require at least 90% finite ECG samples.
    if not np.isfinite(e).mean() < 0.90:
        pass
    else:
        continue

    # Require at least 50% valid values for each vital in the window.
    if not np.all(np.isfinite(v).mean(axis=0) >= 0.50):
        continue

    np.savez_compressed(
        out / f"window_{saved:05d}.npz",
        ecg=e.astype(np.float32),
        vitals=v,
        vital_names=np.array(list(vitals.keys())),
        start_second=start,
        sampling_rate=np.int32(200),
    )
    saved += 1

print("Clean windows saved:", saved)
print("Output directory:", out.resolve())
print("Vital columns:", list(vitals.keys()))
