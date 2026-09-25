from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import TraceSet
from geowave.model.geometry import GeometryIndex

@dataclass(frozen=True)
class GeometryQC:
    trace_count: int
    shot_count: int
    receiver_count: int
    min_offset_m: float
    max_offset_m: float
    median_offset_m: float
    fold_min: int
    fold_max: int

def geometry_qc(dataset: TraceSet, bin_m: float=25) -> GeometryQC:
    offsets = np.array([t.header.offset_m for t in dataset.traces], float)
    geom = GeometryIndex(dataset, bin_m, bin_m)
    fold = list(geom.fold_map().values())
    return GeometryQC(len(dataset.traces), len(dataset.by_shot()), len(dataset.by_receiver()), float(offsets.min()) if offsets.size else 0, float(offsets.max()) if offsets.size else 0, float(np.median(offsets)) if offsets.size else 0, min(fold, default=0), max(fold, default=0))

def azimuth_gaps(dataset: TraceSet, bins: int=18) -> list[tuple[float, float]]:
    a = np.array(GeometryIndex(dataset).azimuths_deg())
    hist, edges = np.histogram(a, bins=np.linspace(0, 360, bins + 1))
    return [(float(edges[i]), float(edges[i + 1])) for i, n in enumerate(hist) if n == 0]
