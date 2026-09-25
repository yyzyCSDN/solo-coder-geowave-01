from __future__ import annotations
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from geowave.model.velocity import VelocityFunction
from geowave.ray.bending import GridVelocity

def build_velocity_grid(functions: list[VelocityFunction], x_m: np.ndarray, z_m: np.ndarray, time_to_depth_scale: float=1000.0, smoothing: float=1.0) -> GridVelocity:
    if not functions:
        raise ValueError('no velocity functions')
    x = np.asarray(x_m, float)
    z = np.asarray(z_m, float)
    points = []
    values = []
    for f in functions:
        for p in f.picks:
            points.append((f.x, p.time_s * time_to_depth_scale))
            values.append(p.velocity_mps)
    xx, zz = np.meshgrid(x, z)
    grid = griddata(np.asarray(points), np.asarray(values), (xx, zz), method='linear')
    nearest = griddata(np.asarray(points), np.asarray(values), (xx, zz), method='nearest')
    grid = np.where(np.isnan(grid), nearest, grid)
    if smoothing > 0:
        grid = gaussian_filter(grid, smoothing)
    return GridVelocity(x, z, grid)

def enforce_vertical_gradient(grid: GridVelocity, minimum_gradient_mps_per_m: float=0.0) -> GridVelocity:
    v = grid.velocity.copy()
    dz = np.diff(grid.z)
    for j in range(v.shape[1]):
        for i in range(1, v.shape[0]):
            floor = v[i - 1, j] + minimum_gradient_mps_per_m * dz[i - 1]
            if v[i, j] < floor:
                v[i, j] = floor
    return GridVelocity(grid.x, grid.z, v)
