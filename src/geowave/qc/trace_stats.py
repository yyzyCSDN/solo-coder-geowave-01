from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import Trace

@dataclass(frozen=True)
class TraceStats:
    trace_id: str
    rms: float
    peak: float
    mean: float
    dead_fraction: float
    clipped_fraction: float

def trace_stats(trace: Trace, dead_epsilon: float=1e-10, clip_quantile: float=0.999) -> TraceStats:
    x = trace.samples
    if not x.size:
        return TraceStats(trace.header.trace_id, 0, 0, 0, 1, 0)
    peak = float(np.max(np.abs(x)))
    threshold = float(np.quantile(np.abs(x), clip_quantile))
    clipped = float(np.mean(np.abs(x) >= threshold)) if peak else 0
    return TraceStats(trace.header.trace_id, float(np.sqrt(np.mean(x * x))), peak, float(np.mean(x)), float(np.mean(np.abs(x) <= dead_epsilon)), clipped)

def outlier_trace_ids(stats: list[TraceStats], z: float=5.0) -> list[str]:
    if len(stats) < 3:
        return []
    vals = np.array([s.rms for s in stats])
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) or 1e-12
    score = 0.6745 * (vals - med) / mad
    return [s.trace_id for s, q in zip(stats, score) if abs(q) > z]
