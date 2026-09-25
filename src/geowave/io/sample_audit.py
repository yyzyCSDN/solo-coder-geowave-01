from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import TraceSet

@dataclass(frozen=True)
class SampleAudit:
    trace_count: int
    unique_dt: tuple[float, ...]
    sample_count_min: int
    sample_count_max: int
    start_time_min: float
    start_time_max: float
    finite_fraction: float

def audit_samples(dataset: TraceSet) -> SampleAudit:
    if not dataset.traces:
        return SampleAudit(0, (), 0, 0, 0.0, 0.0, 1.0)
    dts = tuple(sorted({round(t.header.dt_s, 12) for t in dataset.traces}))
    counts = [t.samples.size for t in dataset.traces]
    starts = [t.header.start_s for t in dataset.traces]
    total = sum(counts)
    finite = sum((int(np.isfinite(t.samples).sum()) for t in dataset.traces))
    return SampleAudit(len(dataset.traces), dts, min(counts), max(counts), min(starts), max(starts), finite / total if total else 1.0)

def invalid_trace_ids(dataset: TraceSet) -> list[str]:
    out = []
    for tr in dataset.traces:
        if not np.all(np.isfinite(tr.samples)):
            out.append(tr.header.trace_id)
        elif tr.samples.size == 0:
            out.append(tr.header.trace_id)
        elif tr.header.dt_s <= 0:
            out.append(tr.header.trace_id)
    return out
