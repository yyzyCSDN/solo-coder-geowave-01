from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.ray.bending import GridVelocity, path_time

@dataclass(frozen=True)
class RayPathQC:
    node_count: int
    length_m: float
    travel_time_s: float
    max_turn_deg: float
    min_velocity_mps: float

def inspect_path(grid: GridVelocity, path: np.ndarray) -> RayPathQC:
    p = np.asarray(path, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
        raise ValueError('path must be N x 2')
    segments = np.diff(p, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    turns = []
    for a, b in zip(segments, segments[1:]):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom > 0:
            c = np.clip(np.dot(a, b) / denom, -1, 1)
            turns.append(float(np.rad2deg(np.arccos(c))))
    velocities = [grid.at(float(x), float(z)) for x, z in p]
    return RayPathQC(len(p), float(np.sum(lengths)), path_time(grid, p), max(turns, default=0.0), min(velocities))
