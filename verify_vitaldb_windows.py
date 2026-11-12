from pathlib import Path
import numpy as np

folder = Path("data/processed/vitaldb/case1")
files = sorted(folder.glob("window_*.npz"))

print("Total windows:", len(files))

if files:
    with np.load(files[0], allow_pickle=False) as d:
        print("ECG shape:", d["ecg"].shape)
        print("Vitals shape:", d["vitals"].shape)
        print("Vital names:", d["vital_names"].tolist())
        print("ECG sampling rate:", int(d["sampling_rate"]))
        print("ECG finite:", bool(np.isfinite(d["ecg"]).all()))
        print("Vital missing values:", int(np.isnan(d["vitals"]).sum()))
