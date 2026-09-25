from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import correlate
from geowave.model.traces import Trace

@dataclass(frozen=True)
class StaticShift:
    trace_id: str
    shift_s: float
    score: float

def apply_static(trace: Trace, shift_s: float) -> Trace:
    shift = shift_s / trace.header.dt_s
    t = np.arange(trace.samples.size)
    src = t - shift
    y = np.interp(src, t, trace.samples, left=0, right=0)
    return trace.copy_with(y, static_s=f'{shift_s:.8g}')

def estimate_shift(reference: Trace, target: Trace, max_shift_s: float) -> StaticShift:
    if abs(reference.header.dt_s - target.header.dt_s) > 1e-12:
        raise ValueError('sampling mismatch')
    n = min(reference.samples.size, target.samples.size)
    maxlag = int(round(max_shift_s / reference.header.dt_s))
    c = correlate(target.samples[:n], reference.samples[:n], mode='full')
    lags = np.arange(-n + 1, n)
    mask = np.abs(lags) <= maxlag
    idx = np.argmax(c[mask])
    lag = lags[mask][idx]
    denom = np.linalg.norm(reference.samples[:n]) * np.linalg.norm(target.samples[:n])
    score = float(c[mask][idx] / denom) if denom else 0.0
    return StaticShift(target.header.trace_id, float(lag * reference.header.dt_s), score)
