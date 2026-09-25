from __future__ import annotations
import numpy as np

def isochron(top_time: np.ndarray, bottom_time: np.ndarray) -> np.ndarray:
    top = np.asarray(top_time, float)
    bottom = np.asarray(bottom_time, float)
    if top.shape != bottom.shape:
        raise ValueError('shape mismatch')
    out = bottom - top
    out[out < 0] = np.nan
    return out

def isopach_from_velocity(top_time: np.ndarray, bottom_time: np.ndarray, interval_velocity_mps: np.ndarray | float) -> np.ndarray:
    dt = isochron(top_time, bottom_time)
    v = np.asarray(interval_velocity_mps, float)
    return 0.5 * dt * v

def gross_rock_volume(thickness_m: np.ndarray, cell_area_m2: float, valid_mask: np.ndarray | None=None) -> float:
    t = np.asarray(thickness_m, float)
    mask = np.isfinite(t)
    if valid_mask is not None:
        mask &= np.asarray(valid_mask, bool)
    return float(np.nansum(t[mask]) * cell_area_m2)
