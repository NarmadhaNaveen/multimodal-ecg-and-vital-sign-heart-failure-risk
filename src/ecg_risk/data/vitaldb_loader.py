import os
import glob
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VitalDBAdapter:
    """
    Adapter for perioperative high-resolution surgical physiological data from VitalDB.
    
    Access & Data Layout:
    - Official repository: https://vitaldb.net
    - VitalDB provides surgical cases containing high-frequency waveforms (SNUADC/Solar8000 ECG,
      arterial blood pressure waveforms) alongside 1-second vital sign tracks.
    - This adapter loads local .vital files or downloaded CSV/HDF5 cohorts,
      performs channel discovery, resamples to target rates, extracts perioperative
      cardiovascular collapse / heart failure indicators, and manages patient partitions.
    """

    def __init__(
        self,
        raw_data_dir: str = "data/raw/vitaldb",
        target_ecg_fs: int = 200,
        target_vital_fs: int = 1,
    ):
        self.raw_data_dir = raw_data_dir
        self.target_ecg_fs = target_ecg_fs
        self.target_vital_fs = target_vital_fs

    def check_access_and_records(self) -> Dict[str, Any]:
        """Checks for local vital files or directories."""
        files = glob.glob(os.path.join(self.raw_data_dir, "*.vital")) + \
                glob.glob(os.path.join(self.raw_data_dir, "*.csv")) + \
                glob.glob(os.path.join(self.raw_data_dir, "*.h5"))
        return {
            "available": len(files) > 0,
            "record_count": len(files),
            "files": files[:10],
            "instructions": (
                "To access VitalDB perioperative data:\n"
                "1. Visit https://vitaldb.net and review the open data license.\n"
                "2. Download individual cases or use the vitaldb Python library:\n"
                "   import vitaldb; vf = vitaldb.VitalFile(caseid, track_names=['ECG_II', 'ART_MBP'])\n"
                f"3. Place downloaded files in: {os.path.abspath(self.raw_data_dir)}"
            )
        }

    def load_case(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Loads a VitalDB file or case extract."""
        if not os.path.exists(file_path):
            logger.error(f"VitalDB case file not found: {file_path}")
            return None

        try:
            if file_path.endswith(".vital"):
                import vitaldb
                vf = vitaldb.VitalFile(file_path)
                track_names = vf.get_track_names()
                # ECG discovery
                ecg_tracks = [t for t in track_names if "ECG" in t.upper()][:2]
                ecg_data = vf.to_numpy(ecg_tracks, interval=1.0 / self.target_ecg_fs)
                # Vital discovery
                vital_keys = ["Solar8000/HR", "Solar8000/ART_SBP", "Solar8000/ART_DBP", "Solar8000/ART_MBP", "Solar8000/PLETH_SPO2", "Solar8000/RR"]
                vital_data = vf.to_numpy([t for t in vital_keys if t in track_names], interval=1.0)
                return {
                    "case_id": os.path.basename(file_path),
                    "ecg": ecg_data.T if ecg_data is not None else None,
                    "vitals": vital_data.T if vital_data is not None else None,
                }
            elif file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
                return {
                    "case_id": os.path.basename(file_path),
                    "df": df,
                }
        except ImportError:
            logger.warning("vitaldb library not installed. Install with `pip install vitaldb`.")
            return None
        except Exception as e:
            logger.error(f"Error loading VitalDB case {file_path}: {e}")
            return None
