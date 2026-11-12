import vitaldb
import numpy as np

vf = vitaldb.VitalFile("data/raw/vitaldb/1.vital")

checks = {
    "Solar8000/HR": (30, 220),
    "Solar8000/ART_SBP": (50, 250),
    "Solar8000/ART_DBP": (20, 150),
    "Solar8000/ART_MBP": (30, 180),
    "Solar8000/PLETH_SPO2": (50, 100),
}

for name, (low, high) in checks.items():
    x = np.asarray(
        vf.to_numpy([name], interval=1.0)
    ).reshape(-1)

    finite = np.isfinite(x)
    valid = finite & (x >= low) & (x <= high)

    print(f"\n{name}")
    print("  Total:", len(x))
    print("  Missing:", int((~finite).sum()))
    print("  Outside screening range:", int((finite & ~((x >= low) & (x <= high))).sum()))
    print("  Within screening range:", int(valid.sum()))
