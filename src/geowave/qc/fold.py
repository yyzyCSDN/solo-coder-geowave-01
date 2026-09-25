from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import TraceSet
from geowave.model.geometry import GeometryIndex

@dataclass(frozen=True)
class FoldSummary:
    populated_bins: int
    mean_fold: float
    p10: float
    p90: float
    holes: int

def fold_summary(dataset: TraceSet, bin_m: float=25, expected_bounds: tuple[int, int] | None=None) -> FoldSummary:
    fold = GeometryIndex(dataset, bin_m, bin_m).fold_map()
    vals = np.array(list(fold.values()), float)
    holes = 0
    if expected_bounds:
        nx, ny = expected_bounds
        holes = max(0, nx * ny - len(fold))
    return FoldSummary(len(fold), float(np.mean(vals)) if vals.size else 0, float(np.percentile(vals, 10)) if vals.size else 0, float(np.percentile(vals, 90)) if vals.size else 0, holes)

def low_fold_bins(dataset: TraceSet, minimum: int, bin_m: float=25) -> list[tuple[int, int, int]]:
    fold = GeometryIndex(dataset, bin_m, bin_m).fold_map()
    return [(i, j, n) for (i, j), n in fold.items() if n < minimum]
