"""GLCM / Haralick features (24-d: 6 props x 4 angles, averaged over d in {1,2})."""
from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops

PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
ANGLE_NAMES = ["0", "45", "90", "135"]
LEVELS = 32


def quantize(img_u8: np.ndarray) -> np.ndarray:
    return (img_u8 // 8).astype(np.uint8)


def extract_glcm(img_u8: np.ndarray) -> np.ndarray:
    g = graycomatrix(quantize(img_u8), distances=[1, 2], angles=ANGLES, levels=LEVELS,
                     symmetric=True, normed=True)
    return np.concatenate([graycoprops(g, p).mean(axis=0) for p in PROPS]).astype(np.float32)


def glcm_matrices(img_u8: np.ndarray, d: int = 1, angle: float = 0.0) -> np.ndarray:
    g = graycomatrix(quantize(img_u8), distances=[d], angles=[angle], levels=LEVELS,
                     symmetric=True, normed=True)
    return g[:, :, 0, 0]


def glcm_feature_names() -> list:
    return [f"GLCM_{p}_{a}" for p in PROPS for a in ANGLE_NAMES]
