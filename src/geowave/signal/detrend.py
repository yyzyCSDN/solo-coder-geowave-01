from __future__ import annotations
import numpy as np
from geowave.model.traces import Trace

def remove_mean(trace: Trace) -> Trace:
    x = trace.samples
    return trace.copy_with(x - (float(np.mean(x)) if x.size else 0.0), detrend='mean')

def remove_linear(trace: Trace) -> Trace:
    x = trace.samples
    if x.size < 2:
        return trace.copy_with(x, detrend='linear')
    t = np.arange(x.size, dtype=float)
    A = np.column_stack([t, np.ones(x.size)])
    slope, intercept = np.linalg.lstsq(A, x, rcond=None)[0]
    return trace.copy_with(x - (slope * t + intercept), detrend='linear')

def robust_detrend(trace: Trace, segments: int=8) -> Trace:
    x = trace.samples
    if x.size < 4:
        return remove_mean(trace)
    edges = np.linspace(0, x.size, segments + 1, dtype=int)
    centers = []
    med = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            centers.append((a + b - 1) / 2)
            med.append(float(np.median(x[a:b])))
    trend = np.interp(np.arange(x.size), centers, med)
    return trace.copy_with(x - trend, detrend='robust')
