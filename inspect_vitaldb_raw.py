import vitaldb
import numpy as np

vf = vitaldb.VitalFile("data/raw/vitaldb/1.vital")

for name in [
    "Solar8000/HR",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/ART_MBP",
    "Solar8000/PLETH_SPO2",
]:
    track = vf.find_track(name)
    print("\nTRACK:", name)
    print("Metadata:", getattr(track, "__dict__", repr(track)))

    raw = vf.get_track_samples(name, None)
    raw = np.asarray(raw).reshape(-1)
    valid = raw[np.isfinite(raw)]

    print("Maximum-resolution sample count:", len(raw))
    print("Finite samples:", len(valid))
    print("First 20 raw values:", raw[:20].tolist())
