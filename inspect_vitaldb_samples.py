import vitaldb

vf = vitaldb.VitalFile("data/raw/vitaldb/1.vital")

for name in [
    "Solar8000/HR",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
]:
    print("\nTRACK:", name)

    result = vf.get_track_samples(name, 1.0)

    print("Return type:", type(result))
    print("Shape:", getattr(result, "shape", "N/A"))
    print("First samples:", repr(result[:10]))
