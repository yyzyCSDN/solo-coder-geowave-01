from __future__ import annotations
import numpy as np

def diffraction_curve(midpoints_m: np.ndarray, apex_x_m: float, apex_time_s: float, velocity_mps: float) -> np.ndarray:
    x = np.asarray(midpoints_m, float)
    return np.sqrt(apex_time_s ** 2 + (2 * (x - apex_x_m) / velocity_mps) ** 2)

def collapse_score(image: np.ndarray, x_index: int, z_index: int, radius: int=3) -> float:
    a = np.asarray(image, float)
    i0 = max(0, z_index - radius)
    i1 = min(a.shape[0], z_index + radius + 1)
    j0 = max(0, x_index - radius)
    j1 = min(a.shape[1], x_index + radius + 1)
    patch = np.abs(a[i0:i1, j0:j1])
    center = abs(a[z_index, x_index])
    return float(center / (np.mean(patch) + 1e-12))
