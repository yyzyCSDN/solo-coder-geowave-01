from __future__ import annotations
import numpy as np
from geowave.model.velocity import VelocityFunction, VelocityModel1D

def vertical_time_to_depth(times_s: np.ndarray, model: VelocityModel1D) -> np.ndarray:
    times = np.asarray(times_s, float)
    maxz = float(model.depths_m[-1])
    zs = np.linspace(0, maxz, 4096)
    twt = np.array([2 * model.one_way_time(float(z), steps=128) for z in zs])
    return np.interp(times, twt, zs, left=0, right=maxz)

def rms_time_to_depth(times_s: np.ndarray, velocity: VelocityFunction) -> np.ndarray:
    times = np.asarray(times_s, float)
    depth = np.zeros_like(times)
    if times.size < 2:
        return depth
    for i in range(1, times.size):
        a = times[i - 1]
        b = times[i]
        mid = (a + b) / 2
        depth[i] = depth[i - 1] + 0.5 * velocity.at(float(mid)) * (b - a)
    return depth
