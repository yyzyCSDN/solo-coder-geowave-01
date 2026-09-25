from __future__ import annotations
import numpy as np
from scipy.interpolate import griddata, Rbf
from geowave.model.horizon import Horizon

def grid_horizon(horizon: Horizon, x: np.ndarray, y: np.ndarray, method: str='linear') -> np.ndarray:
    if len(horizon.picks) < 3:
        raise ValueError('need at least three picks')
    pts = np.array([[p.x, p.y] for p in horizon.picks])
    vals = np.array([p.time_s for p in horizon.picks])
    xx, yy = np.meshgrid(x, y)
    if method in {'linear', 'nearest', 'cubic'}:
        return griddata(pts, vals, (xx, yy), method=method)
    if method == 'rbf':
        f = Rbf(pts[:, 0], pts[:, 1], vals, function='thin_plate')
        return f(xx, yy)
    raise ValueError('unknown gridding method')

def fill_grid_holes(grid: np.ndarray, iterations: int=50) -> np.ndarray:
    a = np.asarray(grid, float).copy()
    mask = np.isnan(a)
    for _ in range(iterations):
        if not np.any(mask):
            break
        updated = False
        for i, j in zip(*np.where(mask)):
            vals = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ii, jj = (i + di, j + dj)
                if 0 <= ii < a.shape[0] and 0 <= jj < a.shape[1] and (not np.isnan(a[ii, jj])):
                    vals.append(a[ii, jj])
            if vals:
                a[i, j] = float(np.mean(vals))
                updated = True
        mask = np.isnan(a)
        if not updated:
            break
    return a
