"""Image loading and the fixed classical preprocessing chain."""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.transform import resize


def load_rgb(path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def demo_chain(rgb: np.ndarray, size: int = 256) -> dict:
    gray = to_gray(rgb)
    resized = resize(gray, (size, size), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
    blur = gaussian_filter(resized, sigma=0.8, truncate=1.25)
    mm = (blur - blur.min()) / (blur.max() - blur.min() + 1e-8)
    u8 = np.rint(mm * 255).astype(np.uint8)
    return dict(rgb=rgb, gray=gray, resized=resized, blur=blur, minmax=mm, uint8=u8)


def preprocess(rgb: np.ndarray, size: int = 256) -> np.ndarray:
    return demo_chain(rgb, size)["uint8"]
