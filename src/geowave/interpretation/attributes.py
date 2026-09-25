from __future__ import annotations
import numpy as np
from scipy.ndimage import sobel, gaussian_filter

def dip_magnitude(section: np.ndarray) -> np.ndarray:
    a = np.asarray(section, float)
    gx = sobel(a, axis=1, mode='nearest')
    gt = sobel(a, axis=0, mode='nearest')
    return np.arctan2(-gx, np.abs(gt) + 1e-09)

def local_similarity(section: np.ndarray, sigma: float=1.5) -> np.ndarray:
    a = np.asarray(section, float)
    left = np.roll(a, 1, axis=1)
    right = np.roll(a, -1, axis=1)
    num = gaussian_filter(a * (left + right) / 2, sigma)
    den = np.sqrt(gaussian_filter(a * a, sigma) * gaussian_filter(((left + right) / 2) ** 2, sigma)) + 1e-09
    return np.clip(num / den, -1, 1)

def edge_strength(section: np.ndarray) -> np.ndarray:
    a = np.asarray(section, float)
    gx = sobel(a, axis=1)
    gy = sobel(a, axis=0)
    return np.hypot(gx, gy)
