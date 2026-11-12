import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Any, Optional


class MultimodalECGDataset(Dataset):
    """
    PyTorch Dataset providing synchronized multimodal observations:
    - ecg: Tensor (C_ecg, T_ecg)
    - vitals: Tensor (C_vital, T_vital)
    - mask: Tensor (C_vital, T_vital) [1 = observed, 0 = missing]
    - label: Tensor (1,) [binary risk outcome]
    - metadata: Dict with patient_id, window_idx, timestamps
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        ecg = torch.as_tensor(item["ecg"], dtype=torch.float32)
        vitals = torch.as_tensor(item["vitals"], dtype=torch.float32)
        mask = torch.as_tensor(item["mask"], dtype=torch.float32)
        label = torch.as_tensor([item["label"]], dtype=torch.float32)

        return {
            "ecg": ecg,
            "vitals": vitals,
            "mask": mask,
            "label": label,
            "patient_id": item.get("patient_id", f"p_{idx:04d}"),
            "window_idx": item.get("window_idx", idx),
        }


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that stacks multimodal tensors and preserves metadata.
    """
    ecg_batch = torch.stack([b["ecg"] for b in batch], dim=0)
    vitals_batch = torch.stack([b["vitals"] for b in batch], dim=0)
    mask_batch = torch.stack([b["mask"] for b in batch], dim=0)
    label_batch = torch.stack([b["label"] for b in batch], dim=0)
    patient_ids = [b["patient_id"] for b in batch]
    window_indices = [b["window_idx"] for b in batch]

    return {
        "ecg": ecg_batch,
        "vitals": vitals_batch,
        "mask": mask_batch,
        "label": label_batch,
        "patient_ids": patient_ids,
        "window_indices": window_indices,
    }
