from __future__ import annotations
import numpy as np
from geowave.well.synthetic import LogCurve

def resample_log(curve: LogCurve, depth_step_m: float) -> LogCurve:
    if depth_step_m <= 0:
        raise ValueError('depth step must be positive')
    start = float(curve.depth_m.min())
    end = float(curve.depth_m.max())
    depth = np.arange(start, end + depth_step_m / 2, depth_step_m)
    values = np.interp(depth, curve.depth_m, curve.values)
    return LogCurve(depth, values, curve.name + '-resampled')

def merge_logs(curves: list[LogCurve], depth_step_m: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not curves:
        return (np.array([]), {})
    start = max((float(c.depth_m.min()) for c in curves))
    end = min((float(c.depth_m.max()) for c in curves))
    if end < start:
        raise ValueError('logs do not overlap in depth')
    depth = np.arange(start, end + depth_step_m / 2, depth_step_m)
    values = {c.name: np.interp(depth, c.depth_m, c.values) for c in curves}
    return (depth, values)

def depth_gaps(curve: LogCurve, expected_step_m: float, factor: float=1.5) -> list[tuple[float, float]]:
    d = np.diff(curve.depth_m)
    return [(float(curve.depth_m[i]), float(curve.depth_m[i + 1])) for i in np.where(d > expected_step_m * factor)[0]]
