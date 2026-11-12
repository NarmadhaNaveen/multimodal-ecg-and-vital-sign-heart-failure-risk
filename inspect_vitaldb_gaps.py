import vitaldb
import numpy as np

vf = vitaldb.VitalFile("data/raw/vitaldb/1.vital")

tracks = [
    "Solar8000/HR",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/ART_MBP",
    "Solar8000/PLETH_SPO2",
]

for name in tracks:
    x = np.asarray(
        vf.to_numpy([name], interval=1.0)
    ).reshape(-1)

    finite = np.isfinite(x)
    valid_values = x[finite]

    print(f"\n{name}")
    print("Total samples:", len(x))
    print("Finite samples:", int(finite.sum()))
    print("Missing percentage:", round(100 * (~finite).mean(), 2))

    # Show the first 30 samples as received from VitalDB.
    print("First 30 samples:", x[:30].tolist())

    if len(valid_values):
        print("First valid values:", valid_values[:10].tolist())
