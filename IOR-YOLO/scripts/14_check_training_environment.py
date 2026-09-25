"""Read-only E01 environment report. No benchmark, model download or training."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path


PLATFORM_LABELS = {"kaggle": "Kaggle", "colab": "Google Colab", "windows": "Windows"}
VERIFIED_PROFILES = {
    "kaggle": {"python": "3.12.13", "torch": "2.10.0+cu128",
                "torchvision": "0.25.0+cu128", "cuda_runtime": "12.8",
                "ultralytics": "8.3.220", "numpy": "2.0.2",
                "opencv_python": "4.13.0.92"},
    # Retained only for the documented backup workstation path.
    "windows": {"python_prefix": "3.11.", "torch": "2.5.1+cu121",
                "torchvision": "0.20.1+cu121", "cuda_runtime": "12.1",
                "ultralytics": "8.3.220"},
}


def detect_execution_platform() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        return "Kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return "Google Colab"
    if platform.system() == "Windows":
        return "Windows"
    return platform.system()


def optional_text(path: str) -> str | None:
    candidate = Path(path)
    return candidate.read_text(encoding="utf-8").strip() if candidate.is_file() else None


def distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def inspect(require_cuda: bool = False, expected_platform: str | None = None) -> dict:
    result = {"python": sys.version.split()[0], "python_executable": sys.executable,
              "os": platform.platform(),
              "torch": None, "torchvision": None, "ultralytics": None,
              "numpy": None, "opencv": None, "opencv_python": None,
              "cuda_available": False, "mps_available": False,
              "cuda_runtime": None, "gpu_name": None, "gpu_memory_bytes": None,
              "gpu_count": 0, "visible_gpu_count": 0, "gpu_inventory": [],
              "selected_accelerator": "cpu",
              "selected_formal_device": "cuda:0",
              "execution_platform": detect_execution_platform(),
              "cloud_provider": None,
              "cloud_session_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
              "container_image_git_commit": optional_text("/etc/git_commit"),
              "container_image_build_date": optional_text("/etc/build_date")}
    if result["execution_platform"] in ("Kaggle", "Google Colab"):
        result["cloud_provider"] = result["execution_platform"]
    try:
        import torch
        import torchvision
        import ultralytics
        import numpy
        import cv2
    except ImportError as exc:
        result["error"] = f"missing dependency: {exc}"
        if require_cuda:
            raise RuntimeError(result["error"]) from exc
        return result
    result.update(torch=torch.__version__, torchvision=torchvision.__version__,
                  ultralytics=ultralytics.__version__, cuda_available=torch.cuda.is_available(),
                  numpy=numpy.__version__, opencv=cv2.__version__,
                  opencv_python=distribution_version("opencv-python", "opencv-python-headless"),
                  cuda_runtime=torch.version.cuda,
                  mps_available=bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
    if result["cuda_available"]:
        result["gpu_count"] = torch.cuda.device_count()
        result["visible_gpu_count"] = result["gpu_count"]
        result["gpu_inventory"] = [
            {"index": index, "name": torch.cuda.get_device_properties(index).name,
             "memory_bytes": torch.cuda.get_device_properties(index).total_memory}
            for index in range(result["gpu_count"])
        ]
        properties = torch.cuda.get_device_properties(0)
        result.update(selected_accelerator="cuda:0", gpu_name=properties.name,
                      gpu_memory_bytes=properties.total_memory)
    elif result["mps_available"]:
        result["selected_accelerator"] = "mps"
    if require_cuda:
        expected_label = PLATFORM_LABELS.get(expected_platform, expected_platform)
        if expected_label and result["execution_platform"] != expected_label:
            raise RuntimeError(
                f"expected execution platform {expected_label}, found {result['execution_platform']}"
            )
        if not result["cuda_available"] or result["selected_accelerator"] != "cuda:0":
            raise RuntimeError(f"E01 formal run requires CUDA device 0: {result}")
        if result["ultralytics"] != "8.3.220":
            raise RuntimeError("Ultralytics differs from the E01 behavior pin")
        if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
            raise RuntimeError("E01 code compatibility requires Python >=3.11,<3.13")
        if expected_platform:
            profile = VERIFIED_PROFILES.get(expected_platform)
            if profile is None:
                raise RuntimeError(f"no reviewed formal runtime profile for {expected_platform}")
            mismatches = []
            for key, expected in profile.items():
                if key == "python_prefix":
                    if not result["python"].startswith(expected):
                        mismatches.append(f"python={result['python']} expected {expected}*")
                elif result.get(key) != expected:
                    mismatches.append(f"{key}={result.get(key)} expected {expected}")
            if mismatches:
                raise RuntimeError("runtime differs from reviewed profile: " + "; ".join(mismatches))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="fail unless the pinned E01 CUDA stack is active")
    parser.add_argument("--platform", choices=tuple(PLATFORM_LABELS),
                        help="when CUDA is required, also require this execution platform")
    args = parser.parse_args()
    print(json.dumps(inspect(args.require_cuda, args.platform), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
