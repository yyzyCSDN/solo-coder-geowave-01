from __future__ import annotations
from dataclasses import dataclass
import math
from geowave.model.horizon import Horizon, HorizonPick

@dataclass(frozen=True)
class HorizonConflict:
    x: float
    y: float
    left_time_s: float
    right_time_s: float
    delta_s: float

def merge_horizons(left: Horizon, right: Horizon, spatial_tolerance_m: float=20.0, time_tolerance_s: float=0.02) -> tuple[Horizon, list[HorizonConflict]]:
    picks = list(left.picks)
    conflicts = []
    for rp in right.picks:
        nearest = left.nearest(rp.x, rp.y)
        if nearest is None:
            picks.append(rp)
            continue
        distance = math.hypot(nearest.x - rp.x, nearest.y - rp.y)
        if distance > spatial_tolerance_m:
            picks.append(rp)
            continue
        delta = rp.time_s - nearest.time_s
        if abs(delta) > time_tolerance_s:
            conflicts.append(HorizonConflict(rp.x, rp.y, nearest.time_s, rp.time_s, delta))
            continue
        weight = max(nearest.quality + rp.quality, 1e-12)
        picks.remove(nearest)
        picks.append(HorizonPick((nearest.x * nearest.quality + rp.x * rp.quality) / weight, (nearest.y * nearest.quality + rp.y * rp.quality) / weight, (nearest.time_s * nearest.quality + rp.time_s * rp.quality) / weight, min(1.0, (nearest.quality + rp.quality) / 2)))
    return (Horizon(left.name + '+' + right.name, picks), conflicts)
