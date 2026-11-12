import vitaldb
import numpy as np
from pathlib import Path
import pandas as pd

vf = vitaldb.VitalFile("data/raw/vitaldb/1.vital")

out = Path("data/processed/vitaldb/case1_resampled")
out.mkdir(parents=True, exist_ok=True)

# ECG at 200 Hz
ecg = np.asarray(
    vf.get_track_samples("SNUADC/ECG_II", 0.005),
    dtype=np.float32
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
    x = np.asarray(
        vf.get_track_samples(name, 1.0),
        dtype=np.float32
    ).reshape(-1)

    # Mask invalid values before carrying readings forward.
    x[(~np.isfinite(x)) | (x < low) | (x > high)] = np.nan

    # Fill only one-second gaps; do not fill longer gaps.
    x = pd.Series(x).ffill(limit=1).to_numpy(dtype=np.float32)
    vitals[key] = x

seconds = min(len(ecg) // 200, *(len(x) for x in vitals.values()))
names = list(vitals)
saved = 0

for start in range(0, seconds - 29, 30):
    end = start + 30
    e = ecg[start * 200:end * 200]
    v = np.column_stack([vitals[n][start:end] for n in names])

    if np.isfinite(e).mean() < 0.90:
        continue

    # Require at least 80% valid data in every vital channel.
    if not np.all(np.isfinite(v).mean(axis=0) >= 0.80):
        continue

    np.savez_compressed(
        out / f"window_{saved:05d}.npz",
        ecg=e,
        vitals=v,
        vital_names=np.array(names),
        start_second=start,
        sampling_rate=np.int32(200),
        case_id="vitaldb_case1",
    )
    saved += 1

print("Resampled windows saved:", saved)
print("Output:", out.resolve())
print("Vital channel names:", names)
