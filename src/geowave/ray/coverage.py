from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RayCoverage:
    counts: np.ndarray
    x_edges: np.ndarray
    z_edges: np.ndarray

def accumulate_ray_coverage(paths: list[np.ndarray], x_edges: np.ndarray, z_edges: np.ndarray) -> RayCoverage:
    counts = np.zeros((len(z_edges) - 1, len(x_edges) - 1), dtype=int)
    for path in paths:
        p = np.asarray(path, float)
        for x, z in p:
            ix = np.searchsorted(x_edges, x, side='right') - 1
            iz = np.searchsorted(z_edges, z, side='right') - 1
            if 0 <= ix < counts.shape[1] and 0 <= iz < counts.shape[0]:
                counts[iz, ix] += 1
    return RayCoverage(counts, np.asarray(x_edges, float), np.asarray(z_edges, float))

def coverage_fraction(coverage: RayCoverage, minimum_hits: int=1) -> float:
    return float(np.mean(coverage.counts >= minimum_hits)) if coverage.counts.size else 0.0

def low_coverage_cells(coverage: RayCoverage, minimum_hits: int) -> list[tuple[int, int, int]]:
    out = []
    for iz, ix in zip(*np.where(coverage.counts < minimum_hits)):
        out.append((int(ix), int(iz), int(coverage.counts[iz, ix])))
    return out
