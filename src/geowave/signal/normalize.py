from __future__ import annotations
import numpy as np
from geowave.model.traces import Trace, TraceSet

def peak_normalize(trace: Trace, target: float=1.0) -> Trace:
    peak = float(np.max(np.abs(trace.samples))) if trace.samples.size else 0
    return trace.copy_with(trace.samples * (target / peak) if peak else trace.samples, normalize='peak')

def rms_normalize(trace: Trace, target: float=1.0) -> Trace:
    rms = float(np.sqrt(np.mean(trace.samples ** 2))) if trace.samples.size else 0
    return trace.copy_with(trace.samples * (target / rms) if rms else trace.samples, normalize='rms')

def surface_consistent_scale(dataset: TraceSet) -> TraceSet:
    shot_amp = {}
    rec_amp = {}
    for k, traces in dataset.by_shot().items():
        shot_amp[k] = np.median([np.sqrt(np.mean(t.samples * t.samples)) for t in traces]) + 1e-12
    for k, traces in dataset.by_receiver().items():
        rec_amp[k] = np.median([np.sqrt(np.mean(t.samples * t.samples)) for t in traces]) + 1e-12
    global_amp = np.median(list(shot_amp.values()) + list(rec_amp.values())) or 1.0
    out = []
    for tr in dataset.traces:
        scale = global_amp / np.sqrt(shot_amp[tr.header.shot_id] * rec_amp[tr.header.receiver_id])
        out.append(tr.copy_with(tr.samples * scale, normalize='surface-consistent'))
    return TraceSet(out, dataset.name + '-normalized')
