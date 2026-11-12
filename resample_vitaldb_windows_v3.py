from pathlib import Path
import numpy as np
import pandas as pd
import vitaldb

INPUT_FILE = Path("data/raw/vitaldb/1.vital")
OUTPUT_DIR = Path("data/processed/vitaldb/case1_resampled_v3")

ECG_FS = 200
WINDOW_SECONDS = 30
MAX_AGE_SECONDS = 3.0
MIN_ECG_FINITE = 0.90
MIN_VITAL_OBSERVED = 0.80

TRACKS = {
    "HR": ("Solar8000/HR", 30, 220),
    "SBP": ("Solar8000/ART_SBP", 40, 250),
    "DBP": ("Solar8000/ART_DBP", 20, 150),
    "MAP": ("Solar8000/ART_MBP", 25, 180),
    "SpO2": ("Solar8000/PLETH_SPO2", 50, 100),
}


def get_records(vf, track_name):
    track = vf.trks.get(track_name)
    if track is None:
        raise ValueError(f"Missing track: {track_name}")

    records = getattr(track, "recs", None)
    if records is None and isinstance(track, dict):
        records = track.get("recs")
    if not records:
        raise ValueError(f"No source records: {track_name}")

    times = np.asarray([r["dt"] for r in records], dtype=np.float64)
    values = np.asarray([r["val"] for r in records], dtype=np.float64)
    return times, values


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vf = vitaldb.VitalFile(str(INPUT_FILE))

    # Use the ECG's recorded duration and exact target sample rate.
    ecg = np.asarray(
        vf.get_track_samples("SNUADC/ECG_II", 1 / ECG_FS),
        dtype=np.float32,
    ).reshape(-1)

    total_seconds = len(ecg) // ECG_FS
    grid_times = vf.dtstart + np.arange(total_seconds, dtype=np.float64)

    values_by_channel = {}
    observed_by_channel = {}
    report = []

    for name, (track_name, low, high) in TRACKS.items():
        source_times, source_values = get_records(vf, track_name)

        valid = (
            np.isfinite(source_values)
            & (source_values >= low)
            & (source_values <= high)
        )
        source_times = source_times[valid]
        source_values = source_values[valid]

        # Sort timestamps before mapping measurements to the 1 Hz grid.
        order = np.argsort(source_times)
        source_times = source_times[order]
        source_values = source_values[order]

        # Latest valid source measurement at or before each grid point.
        indices = np.searchsorted(source_times, grid_times, side="right") - 1
        has_prior = indices >= 0
        safe_indices = np.maximum(indices, 0)

        age = np.full(total_seconds, np.inf, dtype=np.float64)
        age[has_prior] = (
            grid_times[has_prior] - source_times[safe_indices[has_prior]]
        )

        usable = has_prior & (age >= 0) & (age <= MAX_AGE_SECONDS)
        grid_values = np.full(total_seconds, np.nan, dtype=np.float32)
        grid_values[usable] = source_values[safe_indices[usable]]

        # Direct valid observation close to the 1 Hz grid point.
        # This is a separate mask from the usable-value mask.
        direct = np.zeros(total_seconds, dtype=np.uint8)
        nearest = np.searchsorted(source_times, grid_times)
        for offset in (0, -1):
            candidate = nearest + offset
            ok = (candidate >= 0) & (candidate < len(source_times))
            positions = np.flatnonzero(ok)
            if len(positions):
                delta = np.abs(
                    source_times[candidate[positions]] - grid_times[positions]
                )
                matched = positions[delta <= 0.25]
                direct[matched] = 1

        values_by_channel[name] = grid_values
        observed_by_channel[name] = direct

        report.append({
            "channel": name,
            "source_records": len(valid),
            "valid_source_records": int(valid.sum()),
            "usable_fraction": float(usable.mean()),
            "direct_observation_fraction": float(direct.mean()),
            "unusable_fraction": float((~usable).mean()),
            "max_age_seconds": MAX_AGE_SECONDS,
        })

    names = list(TRACKS)
    windows_written = 0
    quality_rows = []

    for start in range(0, total_seconds - WINDOW_SECONDS + 1, WINDOW_SECONDS):
        end = start + WINDOW_SECONDS
        ecg_window = ecg[start * ECG_FS:end * ECG_FS]

        if len(ecg_window) != WINDOW_SECONDS * ECG_FS:
            continue
        if float(np.isfinite(ecg_window).mean()) < MIN_ECG_FINITE:
            continue

        vital_window = np.column_stack([
            values_by_channel[n][start:end] for n in names
        ]).astype(np.float32)

        # Mask used by the model: 1 means a fresh usable value exists.
        usable_mask = np.isfinite(vital_window).astype(np.uint8)

        # Keep the original-observation indicator separately.
        direct_mask = np.column_stack([
            observed_by_channel[n][start:end] for n in names
        ]).astype(np.uint8)

        # Do not reject based on direct sampling cadence. Reject windows
        # only if too much of any channel is genuinely unusable.
        if not np.all(usable_mask.mean(axis=0) >= MIN_VITAL_OBSERVED):
            continue

        path = OUTPUT_DIR / f"window_{windows_written:05d}.npz"
        np.savez_compressed(
            path,
            ecg=ecg_window,
            vitals=vital_window,
            mask=usable_mask,
            direct_observation_mask=direct_mask,
            vital_names=np.array(names),
            start_second=np.int64(start),
            sampling_rate=np.int32(ECG_FS),
            case_id=np.array("1.vital"),
        )

        row = {
            "file": path.name,
            "start_second": start,
            "end_second": end,
            "ecg_finite_fraction": float(np.isfinite(ecg_window).mean()),
        }
        for i, n in enumerate(names):
            row[f"{n}_usable_fraction"] = float(usable_mask[:, i].mean())
            row[f"{n}_direct_fraction"] = float(direct_mask[:, i].mean())
        quality_rows.append(row)
        windows_written += 1

    pd.DataFrame(quality_rows).to_csv(
        OUTPUT_DIR / "quality_report.csv", index=False
    )
    pd.DataFrame(report).to_csv(
        OUTPUT_DIR / "channel_report.csv", index=False
    )

    print("Output:", OUTPUT_DIR.resolve())
    print("Windows written:", windows_written)
    print("Maximum carry-forward age:", MAX_AGE_SECONDS, "seconds")
    print("Mask: 1=fresh usable value, 0=unusable/missing")
    print("Separate direct_observation_mask saved.")
    print("Review channel_report.csv and quality_report.csv.")


if __name__ == "__main__":
    main()
