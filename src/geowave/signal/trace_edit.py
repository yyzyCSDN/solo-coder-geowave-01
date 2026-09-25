from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import Trace, TraceSet

@dataclass(frozen=True)
class TraceEditDecision:
    trace_id: str
    action: str
    reason: str

def zero_bad_windows(trace: Trace, windows: list[tuple[float, float]]) -> Trace:
    y = trace.samples.copy()
    for start_s, end_s in windows:
        if end_s <= start_s:
            continue
        i0 = max(0, int(np.floor((start_s - trace.header.start_s) / trace.header.dt_s)))
        i1 = min(len(y), int(np.ceil((end_s - trace.header.start_s) / trace.header.dt_s)))
        y[i0:i1] = 0.0
    return trace.copy_with(y, edited='window-zero')

def apply_trace_decisions(dataset: TraceSet, decisions: list[TraceEditDecision]) -> TraceSet:
    by_id = {d.trace_id: d for d in decisions}
    out = []
    for tr in dataset.traces:
        d = by_id.get(tr.header.trace_id)
        if d is None or d.action == 'keep':
            out.append(tr)
        elif d.action == 'kill':
            out.append(tr.copy_with(np.zeros_like(tr.samples), edit_reason=d.reason, edited='killed'))
        elif d.action == 'drop':
            continue
        else:
            raise ValueError(f'unknown action {d.action}')
    return TraceSet(out, dataset.name + '-edited')

def dead_trace_decisions(dataset: TraceSet, rms_floor: float) -> list[TraceEditDecision]:
    out = []
    for tr in dataset.traces:
        value = float(np.sqrt(np.mean(tr.samples * tr.samples))) if tr.samples.size else 0.0
        if value < rms_floor:
            out.append(TraceEditDecision(tr.header.trace_id, 'kill', f'rms {value:.6g} below {rms_floor:.6g}'))
    return out
