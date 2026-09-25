from __future__ import annotations
import numpy as np
from scipy.signal import coherence
from geowave.model.traces import Trace

def pair_coherence(a: Trace, b: Trace, nperseg: int=128) -> tuple[np.ndarray, np.ndarray]:
    if abs(a.header.dt_s - b.header.dt_s) > 1e-12:
        raise ValueError('sampling mismatch')
    n = min(a.samples.size, b.samples.size, nperseg)
    f, c = coherence(a.samples, b.samples, fs=1 / a.header.dt_s, nperseg=max(8, n))
    return (f, c)

def window_similarity(a: Trace, b: Trace, start_s: float, end_s: float) -> float:
    dt = a.header.dt_s
    if abs(dt - b.header.dt_s) > 1e-12:
        raise ValueError('sampling mismatch')
    ia = max(0, int(round((start_s - a.header.start_s) / dt)))
    ib = max(0, int(round((start_s - b.header.start_s) / dt)))
    n = max(0, int(round((end_s - start_s) / dt)))
    x = a.samples[ia:ia + n]
    y = b.samples[ib:ib + n]
    m = min(x.size, y.size)
    if m < 2:
        return 0.0
    x = x[:m] - np.mean(x[:m])
    y = y[:m] - np.mean(y[:m])
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / d) if d else 0.0
