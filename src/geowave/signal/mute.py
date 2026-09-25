from __future__ import annotations
import numpy as np
from geowave.model.traces import Trace

def top_mute(trace: Trace, cut_s: float, taper_s: float=0.02) -> Trace:
    t = trace.times
    w = np.ones_like(t)
    w[t < cut_s] = 0
    if taper_s > 0:
        m = (t >= cut_s) & (t < cut_s + taper_s)
        w[m] = 0.5 - 0.5 * np.cos(np.pi * (t[m] - cut_s) / taper_s)
    return trace.copy_with(trace.samples * w, mute=f'top:{cut_s:g}')

def bottom_mute(trace: Trace, cut_s: float, taper_s: float=0.02) -> Trace:
    t = trace.times
    w = np.ones_like(t)
    w[t > cut_s] = 0
    if taper_s > 0:
        m = (t <= cut_s) & (t > cut_s - taper_s)
        w[m] = 0.5 - 0.5 * np.cos(np.pi * (cut_s - t[m]) / taper_s)
    return trace.copy_with(trace.samples * w, mute=f'bottom:{cut_s:g}')

def linear_offset_mute(trace: Trace, intercept_s: float, slope_s_per_m: float, taper_s: float=0.02) -> Trace:
    return top_mute(trace, intercept_s + slope_s_per_m * trace.header.offset_m, taper_s)
