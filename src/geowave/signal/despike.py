from __future__ import annotations
import numpy as np
from scipy.ndimage import median_filter
from geowave.model.traces import Trace

def detect_spikes(trace: Trace, window: int=11, threshold: float=8.0) -> np.ndarray:
    if window < 3 or window % 2 == 0:
        raise ValueError('window must be odd and >= 3')
    x = trace.samples
    baseline = median_filter(x, size=window, mode='reflect')
    residual = x - baseline
    local = median_filter(np.abs(residual), size=window, mode='reflect')
    scale = 1.4826 * local + 1e-12
    return np.abs(residual) > threshold * scale

def repair_spikes(trace: Trace, window: int=11, threshold: float=8.0) -> Trace:
    mask = detect_spikes(trace, window, threshold)
    x = trace.samples.copy()
    clean = np.where(~mask)[0]
    bad = np.where(mask)[0]
    if clean.size >= 2 and bad.size:
        x[bad] = np.interp(bad, clean, x[clean])
    return trace.copy_with(x, despike_count=str(int(mask.sum())))

def burst_regions(mask: np.ndarray, minimum: int=2) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=bool)
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(m):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            if i - start >= minimum:
                out.append((start, i))
            start = None
    if start is not None and len(m) - start >= minimum:
        out.append((start, len(m)))
    return out
