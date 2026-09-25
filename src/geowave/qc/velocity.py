from __future__ import annotations
from dataclasses import dataclass
import math
from geowave.model.velocity import VelocityFunction

@dataclass(frozen=True)
class VelocityIssue:
    kind: str
    time_s: float
    value: float
    detail: str

def validate_velocity(func: VelocityFunction, min_mps: float=1200, max_mps: float=8000, max_jump_mps: float=1000) -> list[VelocityIssue]:
    out = []
    prev = None
    for p in func.picks:
        if p.velocity_mps < min_mps or p.velocity_mps > max_mps:
            out.append(VelocityIssue('range', p.time_s, p.velocity_mps, 'outside configured range'))
        if prev and abs(p.velocity_mps - prev.velocity_mps) > max_jump_mps:
            out.append(VelocityIssue('jump', p.time_s, p.velocity_mps, 'large adjacent velocity change'))
        if p.confidence < 0.2:
            out.append(VelocityIssue('confidence', p.time_s, p.confidence, 'low pick confidence'))
        prev = p
    return out

def lateral_gradient(a: VelocityFunction, b: VelocityFunction, time_s: float) -> float:
    d = math.hypot(a.x - b.x, a.y - b.y)
    return abs(a.at(time_s) - b.at(time_s)) / d if d else 0.0
