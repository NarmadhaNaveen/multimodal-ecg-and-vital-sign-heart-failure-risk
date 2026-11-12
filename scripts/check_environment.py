import sys
import os
import subprocess
import torch
from tabulate import tabulate

# Force UTF-8 encoding for Windows terminals if needed
if sys.platform == "win32" and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def check_nvidia_smi():
    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return f"nvidia-smi not available or failed: {e}"

def main():
    print("=" * 80)
    print("SYSTEM ENVIRONMENT & GPU ACCELERATION AUDIT")
    print("=" * 80)

    py_ver = sys.version.split()[0]
    torch_ver = torch.__version__
    cuda_avail = torch.cuda.is_available()

    table_data = [
        ["Python Version", py_ver, sys.executable],
        ["PyTorch Version", torch_ver, "CUDA-enabled build" if "+cu" in torch_ver else "CPU-only build"],
        ["CUDA Available", str(cuda_avail), "REQUIRED FOR FULL TRAINING"],
    ]

    if cuda_avail:
        dev_name = torch.cuda.get_device_name(0)
        cuda_runtime = torch.version.cuda
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_mb = vram_bytes / (1024 ** 2)
        vram_gb = vram_bytes / (1024 ** 3)
        table_data.extend([
            ["GPU Name", dev_name, "Device 0"],
            ["CUDA Runtime", str(cuda_runtime), "Target Driver Compatible"],
            ["VRAM (Total)", f"{vram_mb:.0f} MB ({vram_gb:.2f} GB)", "RTX 3050 Laptop Tier"],
            ["Allocated Memory", f"{torch.cuda.memory_allocated(0)/(1024**2):.1f} MB", "Baseline idle"],
            ["Reserved Memory", f"{torch.cuda.memory_reserved(0)/(1024**2):.1f} MB", "Baseline cache"],
        ])

    print(tabulate(table_data, headers=["Component", "Status / Value", "Notes"], tablefmt="grid"))

    print("\n--- NVIDIA Driver & Hardware Information (nvidia-smi) ---")
    print(check_nvidia_smi())
    print("=" * 80)

    if not cuda_avail:
        print("\nCRITICAL: CUDA is not available. GPU training cannot proceed.")
        sys.exit(1)
    else:
        print("\nSUCCESS: NVIDIA GPU and CUDA acceleration are verified and ready.")
        sys.exit(0)

if __name__ == "__main__":
    main()
