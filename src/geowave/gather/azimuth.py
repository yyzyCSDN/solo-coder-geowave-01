from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from geowave.model.traces import TraceSet

@dataclass(frozen=True)
class AzimuthBin:
    start_deg: float
    end_deg: float
    trace_ids: tuple[str, ...]

def trace_azimuth(trace) -> float:
    dx = trace.header.receiver_x - trace.header.source_x
    dy = trace.header.receiver_y - trace.header.source_y
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

def azimuth_bins(dataset: TraceSet, width_deg: float=22.5) -> list[AzimuthBin]:
    if width_deg <= 0 or width_deg > 180:
        raise ValueError('invalid azimuth bin width')
    edges = np.arange(0.0, 360.0 + width_deg, width_deg)
    groups = [[] for _ in range(len(edges) - 1)]
    for tr in dataset.traces:
        az = trace_azimuth(tr)
        index = min(int(az // width_deg), len(groups) - 1)
        groups[index].append(tr.header.trace_id)
    return [AzimuthBin(float(a), float(b), tuple(ids)) for a, b, ids in zip(edges[:-1], edges[1:], groups)]

def azimuth_entropy(dataset: TraceSet, width_deg: float=22.5) -> float:
    bins = azimuth_bins(dataset, width_deg)
    counts = np.asarray([len(b.trace_ids) for b in bins], float)
    if counts.sum() == 0:
        return 0.0
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log(p)) / np.log(len(bins)))
