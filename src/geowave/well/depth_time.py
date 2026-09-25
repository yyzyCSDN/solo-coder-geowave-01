from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.well.checkshot import Checkshot

@dataclass(frozen=True)
class DepthTimeModel:
    depth_m: np.ndarray
    two_way_time_s: np.ndarray

    def time_at_depth(self, depth: float) -> float:
        return float(np.interp(depth, self.depth_m, self.two_way_time_s))

    def depth_at_time(self, time_s: float) -> float:
        return float(np.interp(time_s, self.two_way_time_s, self.depth_m))

def build_depth_time(checkshots: list[Checkshot], enforce_monotonic: bool=True) -> DepthTimeModel:
    if not checkshots:
        raise ValueError('empty checkshot list')
    ordered = sorted(checkshots, key=lambda c: c.depth_m)
    depth = np.asarray([c.depth_m for c in ordered], float)
    time = np.asarray([c.two_way_time_s for c in ordered], float)
    if enforce_monotonic:
        time = np.maximum.accumulate(time)
    if np.any(np.diff(depth) <= 0):
        raise ValueError('duplicate or decreasing depth')
    return DepthTimeModel(depth, time)

def drift_curve(model: DepthTimeModel, reference_velocity_mps: float) -> np.ndarray:
    reference = 2 * model.depth_m / reference_velocity_mps
    return model.two_way_time_s - reference
