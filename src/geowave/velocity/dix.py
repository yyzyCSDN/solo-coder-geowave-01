from __future__ import annotations
from dataclasses import dataclass
from geowave.model.velocity import VelocityPick

@dataclass(frozen=True)
class IntervalVelocity:
    top_time_s: float
    bottom_time_s: float
    velocity_mps: float

def dix_interval(picks: list[VelocityPick]) -> list[IntervalVelocity]:
    ps = sorted(picks, key=lambda p: p.time_s)
    out = []
    prev_t = 0.0
    prev_v2t = 0.0
    for p in ps:
        if p.time_s <= prev_t:
            continue
        cur = p.velocity_mps * p.velocity_mps * p.time_s
        val = (cur - prev_v2t) / (p.time_s - prev_t)
        if val <= 0:
            raise ValueError('nonphysical interval velocity')
        out.append(IntervalVelocity(prev_t, p.time_s, val ** 0.5))
        prev_t = p.time_s
        prev_v2t = cur
    return out
