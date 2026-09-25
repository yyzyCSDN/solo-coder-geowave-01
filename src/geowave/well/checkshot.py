from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Checkshot:
    depth_m: float
    two_way_time_s: float

def interval_velocity(checkshots: list[Checkshot]) -> list[tuple[float, float, float]]:
    cs = sorted(checkshots, key=lambda c: c.depth_m)
    out = []
    for a, b in zip(cs, cs[1:]):
        dz = b.depth_m - a.depth_m
        dt = (b.two_way_time_s - a.two_way_time_s) / 2
        if dt <= 0:
            continue
        out.append((a.depth_m, b.depth_m, dz / dt))
    return out

def time_at_depth(checkshots: list[Checkshot], depth_m: float) -> float:
    cs = sorted(checkshots, key=lambda c: c.depth_m)
    if not cs:
        raise ValueError('empty checkshot')
    return float(np.interp(depth_m, [c.depth_m for c in cs], [c.two_way_time_s for c in cs]))
