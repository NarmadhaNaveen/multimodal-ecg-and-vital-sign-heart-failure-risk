import vitaldb

vf = vitaldb.VitalFile("data/raw/vitaldb/1.vital")

for name in [
    "Solar8000/HR",
    "Solar8000/ART_SBP",
    "Solar8000/PLETH_SPO2",
]:
    track = vf.find_track(name)

    print("\nTRACK:", name)
    print("Object type:", type(track))
    print("Attributes:", getattr(track, "__dict__", repr(track)))
