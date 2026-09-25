from __future__ import annotations
from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly
from geowave.model.traces import Trace, TraceHeader

def resample_trace(trace: Trace, new_dt_s: float, max_denominator: int=1000) -> Trace:
    if new_dt_s <= 0:
        raise ValueError('positive dt required')
    ratio = Fraction(trace.header.dt_s / new_dt_s).limit_denominator(max_denominator)
    y = resample_poly(trace.samples, ratio.numerator, ratio.denominator)
    h = TraceHeader(**trace.header.__dict__ | {'dt_s': new_dt_s})
    return Trace(h, y, trace.tags | {'resample': f'{trace.header.dt_s:g}->{new_dt_s:g}'})

def align_start(trace: Trace, new_start_s: float, fill: float=0.0) -> Trace:
    delta = (trace.header.start_s - new_start_s) / trace.header.dt_s
    shift = int(round(delta))
    if abs(delta - shift) > 1e-06:
        raise ValueError('start shift must align to sample grid')
    x = trace.samples
    if shift > 0:
        y = np.concatenate([np.full(shift, fill), x])
    elif shift < 0:
        y = x[min(-shift, x.size):]
    else:
        y = x.copy()
    h = TraceHeader(**trace.header.__dict__ | {'start_s': new_start_s})
    return Trace(h, y, trace.tags | {'aligned': 'true'})
