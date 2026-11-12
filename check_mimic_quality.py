from pathlib import Path
import numpy as np

folder = Path("data/processed/mimic/p000020")
files = sorted(folder.glob("window_*.npy"))[:100]

print("Windows checked:", len(files))

for channel, name in enumerate(["II", "AVF", "ABP", "PAP"]):
    values = np.concatenate([np.load(f)[:, channel] for f in files])
    print(f"\n{name}")
    print("  Min:", round(float(values.min()), 3))
    print("  Max:", round(float(values.max()), 3))
    print("  Mean:", round(float(values.mean()), 3))
    print("  Std:", round(float(values.std()), 3))
    print("  Finite:", bool(np.isfinite(values).all()))
