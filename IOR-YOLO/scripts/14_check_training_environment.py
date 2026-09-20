"""Read-only E01 environment report. No benchmark, model download or training."""

from __future__ import annotations

import argparse
import json
import platform
import sys


def inspect(require_cuda: bool = False) -> dict:
    result = {"python": sys.version.split()[0], "os": platform.platform(),
              "torch": None, "torchvision": None, "ultralytics": None,
              "cuda_available": False, "mps_available": False,
              "cuda_runtime": None, "gpu_name": None, "gpu_memory_bytes": None,
              "selected_accelerator": "cpu"}
    try:
        import torch
        import torchvision
        import ultralytics
    except ImportError as exc:
        result["error"] = f"missing dependency: {exc}"
        if require_cuda:
            raise RuntimeError(result["error"]) from exc
        return result
    result.update(torch=torch.__version__, torchvision=torchvision.__version__,
                  ultralytics=ultralytics.__version__, cuda_available=torch.cuda.is_available(),
                  cuda_runtime=torch.version.cuda,
                  mps_available=bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
    if result["cuda_available"]:
        properties = torch.cuda.get_device_properties(0)
        result.update(selected_accelerator="cuda:0", gpu_name=properties.name,
                      gpu_memory_bytes=properties.total_memory)
    elif result["mps_available"]:
        result["selected_accelerator"] = "mps"
    if require_cuda:
        if (platform.system() != "Windows" or not result["cuda_available"] or
                "RTX 3070" not in (result["gpu_name"] or "")):
            raise RuntimeError(f"E01 formal run requires Windows RTX 3070 CUDA: {result}")
        if not result["torch"].startswith("2.5.1+") or result["torchvision"] != "0.20.1+cu121":
            raise RuntimeError("PyTorch/torchvision CUDA wheels differ from E01 pin")
        if result["cuda_runtime"] != "12.1" or result["ultralytics"] != "8.3.220":
            raise RuntimeError("CUDA runtime or Ultralytics version differs from E01 pin")
        if not result["python"].startswith("3.11."):
            raise RuntimeError("Python must be 3.11.x for formal E01")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="fail unless pinned Windows RTX 3070 stack is active")
    args = parser.parse_args()
    print(json.dumps(inspect(args.require_cuda), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
