"""MVTec AD texture pool, stratified split, masks, dataset stats."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def discover_categories(root) -> list:
    return sorted(p.name for p in Path(root).iterdir() if (p / "train").is_dir())


def build_pool(root, category: str) -> pd.DataFrame:
    base = Path(root) / category
    rows = []
    for origin in ("train", "test"):
        split_dir = base / origin
        for sub in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            dtype = sub.name
            for f in sorted(sub.glob("*.png")):
                good = dtype == "good"
                mask = None if good else base / "ground_truth" / dtype / f"{f.stem}_mask.png"
                rows.append(dict(
                    img_id=f"{category}/{origin}/{dtype}/{f.stem}", path=str(f),
                    mask_path=str(mask) if mask is not None else "", label=0 if good else 1,
                    defect_type=dtype, origin=origin, category=category))
    return pd.DataFrame(rows)


def make_split(pool: pd.DataFrame, seed: int, test_size: float = 0.30):
    train, test = train_test_split(pool, test_size=test_size, stratify=pool["defect_type"],
                                   random_state=seed)
    return train.sort_values("img_id").reset_index(drop=True), test.sort_values("img_id").reset_index(drop=True)


def load_mask(row, size: int) -> np.ndarray:
    """Binary mask (size,size) uint8 {0,1}; all-zero for normal images."""
    if not row["mask_path"]:
        return np.zeros((size, size), np.uint8)
    m = Image.open(row["mask_path"]).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)


def dataset_stats_table(root, categories, seed: int = 42, test_size: float = 0.30) -> pd.DataFrame:
    rows = []
    for c in categories:
        pool = build_pool(root, c)
        tr, te = make_split(pool, seed, test_size)
        r = dict(category=c, n_normal=int((pool.label == 0).sum()), n_defect=int((pool.label == 1).sum()),
                 train_normal=int((tr.label == 0).sum()), train_defect=int((tr.label == 1).sum()),
                 test_normal=int((te.label == 0).sum()), test_defect=int((te.label == 1).sum()))
        for d, n in pool[pool.label == 1].defect_type.value_counts().sort_index().items():
            r[f"n_{d}"] = int(n)
        rows.append(r)
    return pd.DataFrame(rows)


def split_manifest(root, categories, seed: int, test_size: float = 0.30) -> pd.DataFrame:
    parts = []
    for c in categories:
        tr, te = make_split(build_pool(root, c), seed, test_size)
        for name, df in (("train", tr), ("test", te)):
            parts.append(df.assign(split=name)[["img_id", "category", "label", "defect_type", "split"]])
    return pd.concat(parts, ignore_index=True)
