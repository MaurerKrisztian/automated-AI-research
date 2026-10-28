"""Save experiment results together with the environment and command that produced them."""
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def env_info():
    """Versions, device and precision flags of this process, plus the git commit if there is one."""
    cuda = torch.cuda.is_available()
    status = _git("status", "--porcelain")
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if cuda else platform.processor() or "cpu",
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
    }


def save_results(data, out_dir, name):
    """Write {"env": env_info(), **data} to out_dir/<name>_<timestamp>.json and return the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump({"env": env_info(), **data}, f, indent=1)
    return path
