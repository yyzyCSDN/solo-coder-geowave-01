from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import TraceSet

@dataclass(frozen=True)
class RepeatabilityQC:
    matched_traces: int
    median_correlation: float
    median_nrms: float
    unmatched_reference: int
    unmatched_monitor: int

def compare_repeats(reference: TraceSet, monitor: TraceSet) -> RepeatabilityQC:
    left = {t.header.trace_id: t for t in reference.traces}
    right = {t.header.trace_id: t for t in monitor.traces}
    common = sorted(set(left) & set(right))
    correlations = []
    nrms_values = []
    for key in common:
        a = left[key].samples
        b = right[key].samples
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]
        if n < 2:
            continue
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else 0.0
        denom = (np.sqrt(np.mean(a * a)) + np.sqrt(np.mean(b * b))) / 2 + 1e-12
        nrms = float(200 * np.sqrt(np.mean((a - b) ** 2)) / denom)
        correlations.append(corr)
        nrms_values.append(nrms)
    return RepeatabilityQC(len(common), float(np.median(correlations)) if correlations else 0.0, float(np.median(nrms_values)) if nrms_values else 0.0, len(set(left) - set(right)), len(set(right) - set(left)))
