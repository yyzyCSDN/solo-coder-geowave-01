from __future__ import annotations
import numpy as np
from geowave.model.traces import Trace
from geowave.model.velocity import VelocityFunction

def nmo_correct(trace: Trace, velocity: VelocityFunction, stretch_limit: float=0.3) -> Trace:
    dt = trace.header.dt_s
    t0 = trace.times
    offset = trace.header.offset_m
    out = np.zeros_like(trace.samples)
    for i, t in enumerate(t0):
        if t <= 0:
            continue
        v = velocity.at(float(t))
        tn = np.sqrt(t * t + (offset / v) ** 2)
        stretch = (tn - t) / t
        if stretch > stretch_limit:
            continue
        pos = (tn - trace.header.start_s) / dt
        j0 = int(np.floor(pos))
        w = pos - j0
        if 0 <= j0 < trace.samples.size - 1:
            out[i] = (1 - w) * trace.samples[j0] + w * trace.samples[j0 + 1]
    return trace.copy_with(out, nmo='true')

def inverse_nmo_time(t0_s: float, offset_m: float, velocity_mps: float) -> float:
    return float(np.sqrt(max(0.0, t0_s * t0_s + (offset_m / velocity_mps) ** 2)))
