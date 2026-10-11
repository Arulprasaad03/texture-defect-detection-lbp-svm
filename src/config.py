"""Config dataclass + YAML loader. All paths resolved absolute and checked to be on D:."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .utils import assert_on_d


@dataclass
class Config:
    dataset_root: Path
    output_root: Path
    cache_root: Path
    categories: list
    seeds: list
    b2_seeds: list
    image_size: int = 256
    test_size: float = 0.30
    cv_folds: int = 5
    glcm_levels: int = 32
    mi_k: int = 100
    cnn: dict = field(default_factory=dict)
    patch: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    gallery_seed: int = 42
    run_ablations: bool = True
    run_gallery: bool = True


def load_config(path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for k in ("dataset_root", "output_root", "cache_root"):
        raw[k] = assert_on_d(raw[k])
    return Config(**raw)
