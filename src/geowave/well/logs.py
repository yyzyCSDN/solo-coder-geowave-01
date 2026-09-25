from __future__ import annotations
import numpy as np
from scipy.signal import medfilt, savgol_filter
from .synthetic import LogCurve

def despike_log(curve: LogCurve, kernel: int=5, threshold: float=5.0) -> LogCurve:
    if kernel % 2 == 0:
        raise ValueError('kernel must be odd')
    baseline = medfilt(curve.values, kernel_size=kernel)
    residual = curve.values - baseline
    mad = np.median(np.abs(residual - np.median(residual))) + 1e-12
    values = curve.values.copy()
    mask = np.abs(residual) > threshold * 1.4826 * mad
    values[mask] = baseline[mask]
    return LogCurve(curve.depth_m, values, curve.name + '-despiked')

def smooth_log(curve: LogCurve, window: int=11, polyorder: int=2) -> LogCurve:
    if window > len(curve.values):
        window = len(curve.values) if len(curve.values) % 2 else len(curve.values) - 1
    if window < 3:
        return LogCurve(curve.depth_m, curve.values.copy(), curve.name + '-smooth')
    return LogCurve(curve.depth_m, savgol_filter(curve.values, window, min(polyorder, window - 1)), curve.name + '-smooth')

def block_average(curve: LogCurve, block_m: float) -> LogCurve:
    if block_m <= 0:
        raise ValueError('block size must be positive')
    start = float(curve.depth_m.min())
    bins = np.floor((curve.depth_m - start) / block_m).astype(int)
    depths = []
    values = []
    for b in np.unique(bins):
        mask = bins == b
        depths.append(float(np.mean(curve.depth_m[mask])))
        values.append(float(np.mean(curve.values[mask])))
    return LogCurve(np.asarray(depths), np.asarray(values), curve.name + '-blocked')
