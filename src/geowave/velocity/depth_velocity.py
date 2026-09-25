from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.velocity import VelocityFunction

@dataclass(frozen=True)
class DepthVelocityCurve:
    depth_m: np.ndarray
    interval_velocity_mps: np.ndarray
    two_way_time_s: np.ndarray

def rms_to_interval_depth(func: VelocityFunction) -> DepthVelocityCurve:
    picks = func.picks
    if not picks:
        return DepthVelocityCurve(np.array([]), np.array([]), np.array([]))
    times = np.asarray([p.time_s for p in picks], float)
    rms = np.asarray([p.velocity_mps for p in picks], float)
    interval = np.empty_like(rms)
    depth = np.empty_like(rms)
    previous_v2t = 0.0
    previous_t = 0.0
    previous_depth = 0.0
    for i, (t, v) in enumerate(zip(times, rms)):
        if t <= previous_t:
            raise ValueError('times must increase')
        current_v2t = v * v * t
        value = (current_v2t - previous_v2t) / (t - previous_t)
        if value <= 0:
            raise ValueError('nonphysical Dix interval')
        vint = value ** 0.5
        interval[i] = vint
        previous_depth += 0.5 * vint * (t - previous_t)
        depth[i] = previous_depth
        previous_t = t
        previous_v2t = current_v2t
    return DepthVelocityCurve(depth, interval, times)

def average_velocity(curve: DepthVelocityCurve) -> np.ndarray:
    if not len(curve.depth_m):
        return np.array([])
    return np.divide(2 * curve.depth_m, curve.two_way_time_s, out=np.zeros_like(curve.depth_m), where=curve.two_way_time_s > 0)
