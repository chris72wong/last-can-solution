"""Record the actual interpreter/packages/GPU, without contacting the robot."""
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def main():
    info = {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": {}}
    for name in ("torch", "torchvision", "ultralytics", "pillow", "websocket-client", "numpy"):
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = "NOT INSTALLED"
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda"] = torch.version.cuda
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except Exception as exc:
        info["cuda_available"] = False
        info["error"] = f"{type(exc).__name__}: {exc}"
    destination = Path(__file__).resolve().parent.parent / "verification" / "environment.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0 if info["cuda_available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
