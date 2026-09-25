from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .cmp import Gather

@dataclass(frozen=True)
class RadonPanel:
    p_s_per_m: np.ndarray
    times_s: np.ndarray
    amplitudes: np.ndarray

def linear_radon(gather: Gather, p_values: np.ndarray) -> RadonPanel:
    if not gather.traces:
        raise ValueError('empty gather')
    dt = gather.traces[0].header.dt_s
    n = min((t.samples.size for t in gather.traces))
    t0 = gather.traces[0].header.start_s
    panel = np.zeros((len(p_values), n))
    times = t0 + np.arange(n) * dt
    offsets = np.array([tr.header.offset_m for tr in gather.traces])
    for ip, p in enumerate(p_values):
        for tr, off in zip(gather.traces, offsets):
            src = (times + p * off - t0) / dt
            panel[ip] += np.interp(src, np.arange(tr.samples.size), tr.samples, left=0, right=0)
        panel[ip] /= len(gather.traces)
    return RadonPanel(np.asarray(p_values, float), times, panel)
