import sys
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def get_environment_info() -> Dict[str, Any]:
    """Inspect and report the environment details."""
    import torch
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": None,
        "vram_total_mb": 0,
        "vram_allocated_mb": 0,
        "vram_reserved_mb": 0,
    }

    if torch.cuda.is_available():
        dev_idx = 0
        props = torch.cuda.get_device_properties(dev_idx)
        info["device_name"] = props.name
        info["vram_total_mb"] = round(props.total_memory / (1024 ** 2), 2)
        info["vram_allocated_mb"] = round(torch.cuda.memory_allocated(dev_idx) / (1024 ** 2), 2)
        info["vram_reserved_mb"] = round(torch.cuda.memory_reserved(dev_idx) / (1024 ** 2), 2)

    return info


def verify_gpu_or_raise(mandatory: bool = True) -> "torch.device":
    """
    Verifies that CUDA is available and returns the PyTorch GPU device.
    If mandatory is True and CUDA is unavailable, raises a RuntimeError with diagnostics.
    """
    import torch
    if not torch.cuda.is_available():
        diagnostic = (
            "\n" + "=" * 80 + "\n"
            "CRITICAL ERROR: CUDA GPU is not available in the active PyTorch environment.\n"
            "GPU-accelerated training is mandatory for this project.\n"
            f"Active Python: {sys.executable} (Version: {sys.version.split()[0]})\n"
            f"Active PyTorch: {torch.__version__}\n\n"
            "Diagnostic / Remediation Steps:\n"
            "1. Verify that your NVIDIA GPU is visible in Windows by running: nvidia-smi\n"
            "2. Ensure you are running Python within the configured .venv (Python 3.11).\n"
            "3. Install CUDA-enabled PyTorch build using:\n"
            "   .\\.venv\\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "4. Verify in Python:\n"
            "   python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"\n"
            + "=" * 80 + "\n"
        )
        if mandatory:
            logger.error(diagnostic)
            raise RuntimeError(diagnostic)
        else:
            logger.warning("CUDA unavailable. Proceeding on CPU because mandatory=False.")
            return torch.device("cpu")

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    logger.info(f"Verified GPU: {device_name} | VRAM: {vram_mb:.0f} MB | CUDA Runtime: {torch.version.cuda}")
    return device


def log_gpu_memory(prefix: str = "") -> Dict[str, float]:
    """Returns and logs current GPU memory allocations."""
    import torch
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
    alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
    res = torch.cuda.memory_reserved(0) / (1024 ** 2)
    msg = f"{prefix}GPU Memory: {alloc:.1f} MB allocated | {res:.1f} MB reserved"
    logger.info(msg)
    return {"allocated_mb": alloc, "reserved_mb": res}
