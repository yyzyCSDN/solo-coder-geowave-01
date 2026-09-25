from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.ray.bending import GridVelocity

@dataclass(frozen=True)
class VelocityGradientStats:
    max_vertical: float
    max_lateral: float
    median_vertical: float
    median_lateral: float

def gradients(grid: GridVelocity) -> tuple[np.ndarray, np.ndarray]:
    dz = np.gradient(grid.z)
    dx = np.gradient(grid.x)
    gz = np.gradient(grid.velocity, axis=0) / dz[:, None]
    gx = np.gradient(grid.velocity, axis=1) / dx[None, :]
    return (gx, gz)

def gradient_stats(grid: GridVelocity) -> VelocityGradientStats:
    gx, gz = gradients(grid)
    return VelocityGradientStats(float(np.max(np.abs(gz))), float(np.max(np.abs(gx))), float(np.median(np.abs(gz))), float(np.median(np.abs(gx))))

def gradient_anomalies(grid: GridVelocity, lateral_limit: float, vertical_limit: float) -> list[tuple[int, int, str, float]]:
    gx, gz = gradients(grid)
    out = []
    for i, j in zip(*np.where(np.abs(gx) > lateral_limit)):
        out.append((int(i), int(j), 'lateral', float(gx[i, j])))
    for i, j in zip(*np.where(np.abs(gz) > vertical_limit)):
        out.append((int(i), int(j), 'vertical', float(gz[i, j])))
    return out
