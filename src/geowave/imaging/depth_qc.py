from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class DepthConversionQC:
    monotonic_fraction: float
    stretch_p95: float
    squeeze_p05: float
    invalid_columns: int

def depth_conversion_qc(time_axis_s: np.ndarray, depth_grid_m: np.ndarray) -> DepthConversionQC:
    t = np.asarray(time_axis_s, float)
    z = np.asarray(depth_grid_m, float)
    if z.shape[-1] != len(t):
        raise ValueError('last depth dimension must match time axis')
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError('time axis must increase')
    dz = np.diff(z, axis=-1)
    valid = np.all(dz > 0, axis=-1)
    local_velocity = 2.0 * dz / dt
    finite = local_velocity[np.isfinite(local_velocity) & (local_velocity > 0)]
    if finite.size:
        median = np.median(finite)
        stretch = finite / median
        p95 = float(np.percentile(stretch, 95))
        p05 = float(np.percentile(stretch, 5))
    else:
        p95 = 0.0
        p05 = 0.0
    return DepthConversionQC(float(np.mean(valid)), p95, p05, int((~valid).sum()))

def depth_residual(predicted_depth_m: np.ndarray, control_depth_m: np.ndarray) -> dict[str, float]:
    a = np.asarray(predicted_depth_m, float)
    b = np.asarray(control_depth_m, float)
    if a.shape != b.shape:
        raise ValueError('shape mismatch')
    d = a - b
    return {'bias_m': float(np.mean(d)), 'rmse_m': float(np.sqrt(np.mean(d * d))), 'p95_abs_m': float(np.percentile(np.abs(d), 95))}
