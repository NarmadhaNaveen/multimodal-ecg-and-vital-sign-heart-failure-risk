
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb

INPUT_FILE = Path("data/raw/vitaldb/1.vital")
OUTPUT_DIR = Path("data/processed/vitaldb/case1_resampled_masked")

ECG_FS = 200
WINDOW_SECONDS = 30
MIN_ECG_FINITE = 0.90
MIN_VITAL_OBSERVED = 0.80

TRACKS = {
    "HR": ("Solar8000/HR", 30, 220),
    "SBP": ("Solar8000/ART_SBP", 50, 250),
    "DBP": ("Solar8000/ART_DBP", 20, 150),
    "MAP": ("Solar8000/ART_MBP", 30, 180),
    "SpO2": ("Solar8000/PLETH_SPO2", 50, 100),
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vf = vitaldb.VitalFile(str(INPUT_FILE))

    ecg = np.asarray(
        vf.get_track_samples("SNUADC/ECG_II", 1 / ECG_FS),
        dtype=np.float32,
    ).reshape(-1)

    filled_vitals = {}
    observed_masks = {}

    for key, (track_name, low, high) in TRACKS.items():
        x = np.asarray(
            vf.get_track_samples(track_name, 1.0),
            dtype=np.float32,
        ).reshape(-1)

        # Validity is determined BEFORE imputation.
        valid = np.isfinite(x) & (x >= low) & (x <= high)
        observed_masks[key] = valid.astype(np.uint8)

        cleaned = x.copy()
        cleaned[~valid] = np.nan

        # Fill at most one missing second after an observation.
        filled = pd.Series(cleaned).ffill(limit=1).to_numpy(
            dtype=np.float32
        )
        filled_vitals[key] = filled

    names = list(TRACKS)
    seconds = min(
        len(ecg) // ECG_FS,
        *(len(filled_vitals[n]) for n in names),
        *(len(observed_masks[n]) for n in names),
    )

    report = []
    saved = 0

    for start in range(0, seconds - WINDOW_SECONDS + 1, WINDOW_SECONDS):
        end = start + WINDOW_SECONDS

        ecg_window = ecg[start * ECG_FS:end * ECG_FS]

        vital_window = np.column_stack([
            filled_vitals[n][start:end] for n in names
        ]).astype(np.float32)

        mask_window = np.column_stack([
            observed_masks[n][start:end] for n in names
        ]).astype(np.uint8)

        # Keep the original ECG quality rule.
        ecg_finite_fraction = float(np.isfinite(ecg_window).mean())
        if ecg_finite_fraction < MIN_ECG_FINITE:
            continue

        # Preserve the existing post-imputation finite-data filter.
        if not np.all(
            np.isfinite(vital_window).mean(axis=0) >= MIN_VITAL_OBSERVED
        ):
            continue

        # Do not overwrite an existing output window.
        output_file = OUTPUT_DIR / f"window_{saved:05d}.npz"
        if output_file.exists():
            raise FileExistsError(
                f"{output_file} already exists. "
                "Choose a new OUTPUT_DIR to avoid overwriting data."
            )

        np.savez_compressed(
            output_file,
            ecg=ecg_window,
            vitals=vital_window,
            mask=mask_window,
            vital_names=np.array(names),
            start_second=np.int64(start),
            sampling_rate=np.int32(ECG_FS),
            case_id="vitaldb_case1",
        )

        report.append({
            "window_file": output_file.name,
            "case_id": "vitaldb_case1",
            "start_second": start,
            "end_second": end,
            "ecg_finite_fraction": ecg_finite_fraction,
            **{
                f"{n}_observed_fraction": float(mask_window[:, i].mean())
                for i, n in enumerate(names)
            },
            **{
                f"{n}_imputed_or_missing_fraction": float(
                    1 - np.isfinite(vital_window[:, i]).mean()
                )
                for i, n in enumerate(names)
            },
        })

        saved += 1

    report_path = OUTPUT_DIR / "quality_report.csv"
    if report_path.exists():
        raise FileExistsError(
            f"{report_path} already exists. Choose a new OUTPUT_DIR."
        )

    pd.DataFrame(report).to_csv(report_path, index=False)

    print(f"Windows saved: {saved}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Quality report: {report_path.resolve()}")
    print(f"Vital channels: {names}")
    print("Observation masks: saved (1=originally valid, 0=missing/invalid)")


if __name__ == "__main__":
    main()
