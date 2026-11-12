import vitaldb
import numpy as np

vf = vitaldb.VitalFile("data/raw/vitaldb/1.vital")

names = [
    "Solar8000/HR",
    "Solar8000/ART_SBP",
    "Solar8000/PLETH_SPO2",
]

for name in names:
    print("\nTRACK:", name)

    for interval in [1.0, 0.5, None]:
        x = np.asarray(
            vf.get_track_samples(name, interval)
        ).reshape(-1)

        print(
            "Interval:", interval,
            "| Samples:", len(x),
            "| Finite:", int(np.isfinite(x).sum()),
            "| First 12:", x[:12].tolist()
        )
