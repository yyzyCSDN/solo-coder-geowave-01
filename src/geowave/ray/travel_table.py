from __future__ import annotations
from dataclasses import dataclass
import heapq
import numpy as np
from geowave.ray.bending import GridVelocity

@dataclass(frozen=True)
class TravelTimeTable:
    x: np.ndarray
    z: np.ndarray
    time_s: np.ndarray
    source_index: tuple[int, int]

def fast_grid_travel_times(grid: GridVelocity, source_x: float, source_z: float) -> TravelTimeTable:
    ix = int(np.argmin(np.abs(grid.x - source_x)))
    iz = int(np.argmin(np.abs(grid.z - source_z)))
    dist = np.full(grid.velocity.shape, np.inf)
    dist[iz, ix] = 0.0
    heap = [(0.0, iz, ix)]
    while heap:
        t, i, j = heapq.heappop(heap)
        if t != dist[i, j]:
            continue
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = (i + di, j + dj)
            if not 0 <= ni < len(grid.z) or not 0 <= nj < len(grid.x):
                continue
            step = abs(grid.z[ni] - grid.z[i]) if di else abs(grid.x[nj] - grid.x[j])
            v = 0.5 * (grid.velocity[i, j] + grid.velocity[ni, nj])
            nt = t + step / v
            if nt < dist[ni, nj]:
                dist[ni, nj] = nt
                heapq.heappush(heap, (nt, ni, nj))
    return TravelTimeTable(grid.x.copy(), grid.z.copy(), dist, (iz, ix))

def sample_table(table: TravelTimeTable, x: float, z: float) -> float:
    ix = int(np.argmin(np.abs(table.x - x)))
    iz = int(np.argmin(np.abs(table.z - z)))
    return float(table.time_s[iz, ix])
