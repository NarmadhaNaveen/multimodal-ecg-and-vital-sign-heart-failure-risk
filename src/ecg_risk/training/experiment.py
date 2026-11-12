import os
import json
import csv
import time
from typing import Dict, Any, List


class ExperimentTracker:
    """
    Local experiment tracking system.
    Logs per-epoch metrics, timing, GPU memory consumption, and exports experiment artifacts.
    """

    def __init__(self, output_dir: str = "outputs", experiment_name: str = "multimodal_hf"):
        self.output_dir = output_dir
        self.exp_name = experiment_name
        self.exp_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.metrics_history: List[Dict[str, Any]] = []
        self.csv_path = os.path.join(self.exp_dir, "metrics.csv")
        self.json_path = os.path.join(self.exp_dir, "experiment_summary.json")

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]):
        entry = {"epoch": epoch, "timestamp": time.time(), **metrics}
        self.metrics_history.append(entry)

        # Write to CSV
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(entry.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)

    def finalize(self, summary_data: Dict[str, Any]):
        summary = {
            "experiment_name": self.exp_name,
            "final_metrics": summary_data,
            "num_epochs": len(self.metrics_history),
            "history": self.metrics_history,
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
