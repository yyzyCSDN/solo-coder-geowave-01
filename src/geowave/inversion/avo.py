from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class AVOFit:
    intercept: float
    gradient: float
    curvature: float | None
    r2: float

def shuey_basis(angles_deg: np.ndarray, order: int=2) -> np.ndarray:
    th = np.deg2rad(np.asarray(angles_deg, float))
    s2 = np.sin(th) ** 2
    if order == 2:
        return np.column_stack([np.ones_like(th), s2])
    if order == 3:
        return np.column_stack([np.ones_like(th), s2, np.tan(th) ** 2 - s2])
    raise ValueError('order must be 2 or 3')

def fit_avo(angles_deg: np.ndarray, amplitudes: np.ndarray, weights: np.ndarray | None=None, order: int=2) -> AVOFit:
    X = shuey_basis(angles_deg, order)
    y = np.asarray(amplitudes, float)
    if y.size != X.shape[0]:
        raise ValueError('shape mismatch')
    if weights is not None:
        sw = np.sqrt(np.asarray(weights, float))
        X = X * sw[:, None]
        y = y * sw
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ coef
    ssr = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ssr / sst if sst > 0 else 1.0
    return AVOFit(float(coef[0]), float(coef[1]), float(coef[2]) if order == 3 else None, r2)
