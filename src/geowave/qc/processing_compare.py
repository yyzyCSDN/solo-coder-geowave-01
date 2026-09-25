from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import TraceSet

@dataclass(frozen=True)
class ProcessingComparison:
    matched: int
    median_correlation: float
    median_rms_ratio: float
    changed_trace_fraction: float

def compare_datasets(before: TraceSet, after: TraceSet, correlation_floor: float=0.95) -> ProcessingComparison:
    a = {t.header.trace_id: t for t in before.traces}
    b = {t.header.trace_id: t for t in after.traces}
    common = sorted(set(a) & set(b))
    corr = []
    ratios = []
    changed = 0
    for key in common:
        x = a[key].samples
        y = b[key].samples
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
        c = float(np.corrcoef(x, y)[0, 1]) if n > 1 and np.std(x) and np.std(y) else 0.0
        corr.append(c)
        rx = float(np.sqrt(np.mean(x * x))) + 1e-12
        ry = float(np.sqrt(np.mean(y * y)))
        ratios.append(ry / rx)
        if c < correlation_floor:
            changed += 1
    return ProcessingComparison(len(common), float(np.median(corr)) if corr else 0.0, float(np.median(ratios)) if ratios else 0.0, changed / len(common) if common else 0.0)
