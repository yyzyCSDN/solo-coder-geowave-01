from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import Trace

@dataclass(frozen=True)
class FirstBreakPick:
    trace_id: str
    time_s: float
    confidence: float
    method: str

def sta_lta_pick(trace: Trace, sta_s: float=0.01, lta_s: float=0.1, threshold: float=3.0) -> FirstBreakPick | None:
    dt = trace.header.dt_s
    sta = max(1, int(sta_s / dt))
    lta = max(sta + 1, int(lta_s / dt))
    e = trace.samples * trace.samples
    cs = np.concatenate([[0.0], np.cumsum(e)])
    best_ratio = 0.0
    best_index = -1
    for i in range(lta, len(e)):
        short = (cs[i + 1] - cs[i + 1 - sta]) / sta
        long = (cs[i + 1 - sta] - cs[i + 1 - lta]) / max(1, lta - sta)
        ratio = short / (long + 1e-12)
        if ratio > threshold:
            best_ratio = ratio
            best_index = i
            break
    if best_index < 0:
        return None
    time_s = trace.header.start_s + best_index * dt
    confidence = min(1.0, (best_ratio - threshold) / max(threshold, 1e-09))
    return FirstBreakPick(trace.header.trace_id, time_s, confidence, 'sta-lta')

def aic_pick(trace: Trace, start_s: float | None=None, end_s: float | None=None) -> FirstBreakPick | None:
    x = trace.samples
    i0 = 1 if start_s is None else max(1, int((start_s - trace.header.start_s) / trace.header.dt_s))
    i1 = len(x) - 2 if end_s is None else min(len(x) - 2, int((end_s - trace.header.start_s) / trace.header.dt_s))
    if i1 <= i0:
        return None
    scores = []
    indices = range(i0, i1 + 1)
    for k in indices:
        left = np.var(x[:k]) + 1e-12
        right = np.var(x[k:]) + 1e-12
        scores.append(k * np.log(left) + (len(x) - k - 1) * np.log(right))
    k = i0 + int(np.argmin(scores))
    contrast = float(np.std(x[k:]) / (np.std(x[:k]) + 1e-12))
    return FirstBreakPick(trace.header.trace_id, trace.header.start_s + k * trace.header.dt_s, min(1.0, contrast / 10), 'aic')
