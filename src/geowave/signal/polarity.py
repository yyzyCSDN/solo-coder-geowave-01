from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import Trace

@dataclass(frozen=True)
class PolarityCheck:
    correlation: float
    inverted_correlation: float
    recommended_flip: bool

def compare_polarity(trace: Trace, reference: np.ndarray, start_s: float, end_s: float) -> PolarityCheck:
    i0 = max(0, int((start_s - trace.header.start_s) / trace.header.dt_s))
    i1 = min(trace.samples.size, int((end_s - trace.header.start_s) / trace.header.dt_s))
    x = trace.samples[i0:i1]
    y = np.asarray(reference, float)[:len(x)]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return PolarityCheck(0.0, 0.0, False)
    corr = float(np.corrcoef(x, y)[0, 1])
    inv = float(np.corrcoef(-x, y)[0, 1])
    return PolarityCheck(corr, inv, inv > corr)

def flip_polarity(trace: Trace) -> Trace:
    return trace.copy_with(-trace.samples, polarity='flipped')

def polarity_balance(traces: list[Trace], reference_index: int=0) -> dict[str, int]:
    if not traces:
        return {'same': 0, 'opposite': 0, 'unknown': 0}
    ref = traces[reference_index].samples
    result = {'same': 0, 'opposite': 0, 'unknown': 0}
    for tr in traces:
        n = min(len(ref), len(tr.samples))
        if n < 2 or np.std(ref[:n]) == 0 or np.std(tr.samples[:n]) == 0:
            result['unknown'] += 1
            continue
        c = float(np.corrcoef(ref[:n], tr.samples[:n])[0, 1])
        result['same' if c >= 0 else 'opposite'] += 1
    return result
