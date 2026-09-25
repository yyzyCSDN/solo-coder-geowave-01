from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ResidualMoveoutFit:
    intercept_s: float
    curvature_s_per_m2: float
    r2: float

def fit_residual_moveout(offsets_m: np.ndarray, pick_times_s: np.ndarray) -> ResidualMoveoutFit:
    x = np.asarray(offsets_m, float)
    t = np.asarray(pick_times_s, float)
    if x.size != t.size or x.size < 3:
        raise ValueError('need three or more picks')
    X = np.column_stack([np.ones_like(x), x * x])
    coef = np.linalg.lstsq(X, t, rcond=None)[0]
    pred = X @ coef
    sse = float(np.sum((t - pred) ** 2))
    sst = float(np.sum((t - np.mean(t)) ** 2))
    return ResidualMoveoutFit(float(coef[0]), float(coef[1]), 1.0 - sse / sst if sst else 1.0)

def velocity_scale_from_curvature(t0_s: float, velocity_mps: float, curvature_s_per_m2: float) -> float:
    if t0_s <= 0 or velocity_mps <= 0:
        raise ValueError('positive t0 and velocity required')
    base = 1.0 / (2.0 * t0_s * velocity_mps * velocity_mps)
    ratio = 1.0 + curvature_s_per_m2 / (base + 1e-18)
    return float(1.0 / np.sqrt(max(ratio, 0.1)))
