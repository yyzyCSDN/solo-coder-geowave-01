from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import TraceSet

@dataclass(frozen=True)
class AcquisitionQC:
    shot_spacing_median: float
    receiver_spacing_median: float
    source_gap_max: float
    receiver_gap_max: float
    duplicate_source_positions: int
    duplicate_receiver_positions: int

def _position_stats(points: list[tuple[float, float]]) -> tuple[float, float, int]:
    unique = sorted(set(points))
    duplicates = len(points) - len(unique)
    if len(unique) < 2:
        return (0.0, 0.0, duplicates)
    x = np.asarray([p[0] for p in unique], float)
    y = np.asarray([p[1] for p in unique], float)
    order = np.lexsort((y, x))
    xy = np.column_stack([x[order], y[order]])
    gaps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return (float(np.median(gaps)), float(np.max(gaps)), duplicates)

def acquisition_qc(dataset: TraceSet) -> AcquisitionQC:
    sources = [(t.header.source_x, t.header.source_y) for t in dataset.traces]
    receivers = [(t.header.receiver_x, t.header.receiver_y) for t in dataset.traces]
    ss, sg, sd = _position_stats(sources)
    rs, rg, rd = _position_stats(receivers)
    return AcquisitionQC(ss, rs, sg, rg, sd, rd)
