"""Shared helpers: seeding, D:-drive guards, logging, cache paths, figure saving."""
from __future__ import annotations

import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_OUTPUT_ROOT: Path | None = None
_FOOTER: str = ""


def assert_on_d(path) -> Path:
    """Raise RuntimeError if the resolved absolute path is on C:."""
    p = Path(os.path.abspath(str(path)))
    if str(p).upper().startswith("C:"):
        raise RuntimeError(f"Path is on C: which is forbidden: {p}")
    return p


def check_environment() -> None:
    """Abort with a clear message if any temp/cache location or the interpreter is on C:."""
    names = ["TEMP", "TMP", "MPLCONFIGDIR", "TORCH_HOME", "JOBLIB_TEMP_FOLDER", "PIP_CACHE_DIR"]
    bad = [f"{n}={os.environ.get(n, '<unset>')}" for n in names
           if not os.environ.get(n, "").upper().startswith("D:")]
    if not sys.prefix.upper().startswith("D:"):
        bad.append(f"sys.prefix={sys.prefix}")
    if bad:
        raise SystemExit("ENVIRONMENT ERROR (must be on D:): " + "; ".join(bad)
                         + "\nRun:  . .\\scripts\\env_d.ps1  and activate .venv")
    assert_on_d(tempfile.gettempdir())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def get_logger(output_root, name: str = "run") -> logging.Logger:
    log_dir = assert_on_d(Path(output_root) / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
        fh = logging.FileHandler(log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8")
        sh = logging.StreamHandler()
        for h in (fh, sh):
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger


def set_output_root(path, footer: str = "") -> None:
    global _OUTPUT_ROOT, _FOOTER
    _OUTPUT_ROOT = assert_on_d(path)
    _FOOTER = footer
    for sub in ("figures", "figures_data", "tables", "models", "gallery", "logs"):
        (_OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def output_root() -> Path:
    if _OUTPUT_ROOT is None:
        raise RuntimeError("set_output_root() not called")
    return _OUTPUT_ROOT


def cache_path(cache_root, category: str, kind: str, size: int) -> Path:
    root = assert_on_d(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{category}_{kind}_{size}.npz"


def save_fig(fig, name: str) -> Path:
    import matplotlib.pyplot as plt
    out = output_root() / "figures" / (name if name.endswith(".png") else name + ".png")
    if _FOOTER:
        fig.text(0.99, 0.005, _FOOTER, ha="right", va="bottom", fontsize=7, color="grey")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def save_csv_for_fig(df, name: str) -> Path:
    out = output_root() / "figures_data" / (name if name.endswith(".csv") else name + ".csv")
    df.to_csv(out, index=False, float_format="%.6f")
    return out


def cpu_name() -> str:
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor" + "\\0")
        return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:
        import platform
        return platform.processor() or "unknown CPU"
