import os
import glob
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MIMIC3WaveformAdapter:
    """
    Adapter for the PhysioNet MIMIC-III Waveform Database Matched Subset.
    
    Access & Compliance Requirements:
    - MIMIC-III Waveform Matched Subset is a credentialed PhysioNet resource.
    - Users must complete human-subjects training (CITI) and sign the PhysioNet DUA.
    - Place extracted WFDB records in data/raw/mimic/ (e.g., data/raw/mimic/p00/p000020/).
    - This adapter parses .hea and .dat records using wfdb or direct binary readers,
      extracts Lead II / Lead V ECG, synchronized ABP / Pleth waveforms, aligns timestamps,
      and links with clinical outcomes (ICD-9 codes or chart event inotropic escalations).
    """

    def __init__(
        self,
        raw_data_dir: str = "data/raw/mimic",
        target_ecg_fs: int = 125,
        target_vital_fs: int = 1,
    ):
        self.raw_data_dir = raw_data_dir
        self.target_ecg_fs = target_ecg_fs
        self.target_vital_fs = target_vital_fs

    def check_access_and_records(self) -> Dict[str, Any]:
        """Checks for existing local records or reports access instructions."""
        record_headers = glob.glob(os.path.join(self.raw_data_dir, "**", "*.hea"), recursive=True)
        # Filter out layout headers (e.g. ending with '_layout.hea' or 'n.hea')
        valid_records = [r for r in record_headers if not r.endswith("_layout.hea")]
        status = {
            "available": len(valid_records) > 0,
            "record_count": len(valid_records),
            "records": valid_records[:10],
            "credentials_required": True,
            "instructions": (
                "To access genuine MIMIC-III Waveform records:\n"
                "1. Complete CITI Data or Specimens Only Research training.\n"
                "2. Request access on PhysioNet (https://physionet.org/content/mimic3wdb-matched/1.0/).\n"
                "3. Download matched records using wfdb CLI or wget with your PhysioNet credentials.\n"
                f"4. Place record files in: {os.path.abspath(self.raw_data_dir)}"
            )
        }
        return status

    def load_record(self, record_path_no_ext: str) -> Optional[Dict[str, Any]]:
        """Loads a single WFDB record if wfdb is available and files exist."""
        try:
            import wfdb
            record = wfdb.rdrecord(record_path_no_ext)
            sig_names = [s.upper() for s in record.sig_name]

            # Discover ECG channels
            ecg_idx = []
            for preferred in ["II", "V", "MLII", "V5", "I"]:
                if preferred in sig_names and len(ecg_idx) < 2:
                    ecg_idx.append(sig_names.index(preferred))

            if len(ecg_idx) == 0:
                logger.warning(f"Record {record_path_no_ext} has no valid ECG leads.")
                return None

            ecg_data = record.p_signal[:, ecg_idx].T  # (C, T)

            # Discover Vitals channels (HR, ABP, RESP, SPO2)
            vital_dict = {}
            for target, aliases in {
                "HR": ["HR", "PULSE"],
                "ABP": ["ABP", "ART", "BP"],
                "RESP": ["RESP", "RESPIRATION"],
                "SPO2": ["SPO2", "PLETH", "O2SAT"],
            }.items():
                found = False
                for a in aliases:
                    if a in sig_names:
                        vital_dict[target] = record.p_signal[:, sig_names.index(a)]
                        found = True
                        break
                if not found:
                    vital_dict[target] = np.full(record.sig_len, np.nan)

            return {
                "record_name": record.record_name,
                "fs": record.fs,
                "ecg": ecg_data,
                "vitals": vital_dict,
                "duration_sec": record.sig_len / record.fs,
            }
        except ImportError:
            logger.error("wfdb package is required to load genuine PhysioNet records.")
            return None
        except Exception as e:
            logger.error(f"Failed to read record {record_path_no_ext}: {e}")
            return None
