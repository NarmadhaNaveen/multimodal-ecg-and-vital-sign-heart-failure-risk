import vitaldb
import numpy as np

path = "data/raw/vitaldb/1.vital"
vf = vitaldb.VitalFile(path)

ecg_tracks = [
    "SNUADC/ECG_II",
    "SNUADC/ECG_V5",
]

vital_tracks = [
    "Solar8000/HR",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/ART_MBP",
    "Solar8000/PLETH_SPO2",
]

for name in ecg_tracks + vital_tracks:
    if name not in vf.get_track_names():
        print(f"{name}: NOT FOUND")
        continue

    interval = 0.005 if name in ecg_tracks else 1.0
    data = vf.to_numpy([name], interval=interval)
    x = np.asarray(data).reshape(-1)
    valid = x[np.isfinite(x)]

    print(f"\n{name}")
    print("  Samples:", len(x))
    print("  Valid:", len(valid))
    print("  Missing/non-finite:", len(x) - len(valid))

    if len(valid):
        print("  Min:", round(float(valid.min()), 3))
        print("  Max:", round(float(valid.max()), 3))
        print("  Mean:", round(float(valid.mean()), 3))
