import wfdb
import numpy as np
from pathlib import Path

record_path = r"data/raw/mimic/p00/p000020/3544749_0005"
output_dir = Path("data/processed/mimic/p000020")
output_dir.mkdir(parents=True, exist_ok=True)

record = wfdb.rdrecord(record_path)
fs = int(record.fs)
window_seconds = 30
window_size = fs * window_seconds

print("Sampling rate:", fs)
print("Channels:", record.sig_name)

# Read in chunks so the complete waveform is not held in memory.
chunk_seconds = 300
chunk_size = fs * chunk_seconds
window_count = 0

for start in range(0, record.sig_len, chunk_size):
    end = min(start + chunk_size, record.sig_len)
    chunk = wfdb.rdrecord(
        record_path,
        sampfrom=start,
        sampto=end,
        physical=True
    ).p_signal

    usable = (len(chunk) // window_size) * window_size
    if usable == 0:
        continue

    windows = chunk[:usable].reshape(-1, window_size, len(record.sig_name))

    for window in windows:
        # Skip windows containing NaN or infinite values.
        if not np.isfinite(window).all():
            continue

        np.save(
            output_dir / f"window_{window_count:06d}.npy",
            window.astype(np.float32)
        )
        window_count += 1

print("Saved windows:", window_count)
print("Window duration:", window_seconds, "seconds")
print("Output directory:", output_dir.resolve())
