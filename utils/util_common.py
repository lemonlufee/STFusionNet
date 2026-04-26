import os
import time
import json
import random
import socket
import platform
import subprocess
import numpy as np
import torch
from pathlib import Path

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def to_jsonable(obj):
    """Recursively convert common numpy/torch types to JSON-serializable Python types."""
    # numpy scalar
    if isinstance(obj, np.generic):
        return obj.item()
    # torch tensor
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    # dict
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    # set
    if isinstance(obj, set):
        return [to_jsonable(v) for v in sorted(obj)]
    # numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_stdio_for_server() -> None:
    """Avoid UnicodeEncodeError on headless servers/locales (e.g., C/GBK)."""
    import sys

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def collect_runtime_env() -> dict:
    """Collect lightweight runtime metadata for reproducibility/debugging."""
    py_ver = platform.python_version()
    info = {
        "timestamp": now_str(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": py_ver,
        "cwd": os.getcwd(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        try:
            info["gpu_name_0"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

    # best-effort git metadata
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["git_commit"] = commit
    except Exception:
        info["git_commit"] = None

    return info
