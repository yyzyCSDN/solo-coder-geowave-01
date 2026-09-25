from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter1d
from geowave.model.traces import Trace

def rms_agc(trace: Trace, window_s: float, epsilon: float=1e-09) -> Trace:
    n = max(1, int(round(window_s / trace.header.dt_s)))
    power = uniform_filter1d(trace.samples * trace.samples, size=n, mode='nearest')
    gain = 1.0 / np.sqrt(np.maximum(power, epsilon))
    return trace.copy_with(trace.samples * gain, agc=f'rms:{window_s:g}')

def percentile_agc(trace: Trace, window_s: float, q: float=90.0) -> Trace:
    n = max(3, int(round(window_s / trace.header.dt_s)))
    half = n // 2
    x = trace.samples
    out = np.empty_like(x)
    for i in range(x.size):
        a = max(0, i - half)
        b = min(x.size, i + half + 1)
        scale = np.percentile(np.abs(x[a:b]), q)
        out[i] = x[i] / scale if scale > 1e-12 else 0.0
    return trace.copy_with(out, agc=f'p{q:g}:{window_s:g}')
