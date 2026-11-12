import vitaldb

vf = vitaldb.VitalFile("data/raw/vitaldb/1.vital")

for name in [
    "Solar8000/HR",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
]:
    track = vf.find_track(name)
    print("\nTRACK:", name)
    print("Type:", type(track))
    print("Metadata:", getattr(track, "__dict__", repr(track)))
