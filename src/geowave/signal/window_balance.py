from __future__ import annotations
import numpy as np
from geowave.model.traces import Trace

def window_rms(trace: Trace, windows: list[tuple[float, float]]) -> list[float]:
    values = []
    for start, end in windows:
        i0 = max(0, int((start - trace.header.start_s) / trace.header.dt_s))
        i1 = min(trace.samples.size, int((end - trace.header.start_s) / trace.header.dt_s))
        x = trace.samples[i0:i1]
        values.append(float(np.sqrt(np.mean(x * x))) if x.size else 0.0)
    return values

def balance_windows(trace: Trace, windows: list[tuple[float, float]], target: float | None=None, maximum_gain: float=8.0) -> Trace:
    if not windows:
        return trace.copy_with(trace.samples, window_balance='none')
    rms = window_rms(trace, windows)
    nonzero = [x for x in rms if x > 0]
    target = float(np.median(nonzero)) if target is None and nonzero else float(target or 1.0)
    y = trace.samples.copy()
    for (start, end), value in zip(windows, rms):
        if value <= 0:
            continue
        gain = min(maximum_gain, target / value)
        i0 = max(0, int((start - trace.header.start_s) / trace.header.dt_s))
        i1 = min(trace.samples.size, int((end - trace.header.start_s) / trace.header.dt_s))
        y[i0:i1] *= gain
    return trace.copy_with(y, window_balance=f'target={target:g}')

def gain_curve(trace: Trace, windows: list[tuple[float, float]], target: float | None=None) -> np.ndarray:
    g = np.ones(trace.samples.size)
    rms = window_rms(trace, windows)
    nonzero = [x for x in rms if x > 0]
    desired = float(np.median(nonzero)) if target is None and nonzero else float(target or 1.0)
    for (start, end), value in zip(windows, rms):
        if value <= 0:
            continue
        i0 = max(0, int((start - trace.header.start_s) / trace.header.dt_s))
        i1 = min(trace.samples.size, int((end - trace.header.start_s) / trace.header.dt_s))
        g[i0:i1] = desired / value
    return g
