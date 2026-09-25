from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Layer:
    thickness_m: float
    velocity_mps: float

@dataclass(frozen=True)
class RefractionResult:
    ray_parameter: float
    offset_m: float
    time_s: float
    angles_rad: tuple[float, ...]

def _offset_time(layers: list[Layer], p: float) -> tuple[float, float, list[float]]:
    x = t = 0.0
    angles = []
    for lay in layers:
        s = p * lay.velocity_mps
        if abs(s) >= 1:
            return (float('inf'), float('inf'), [])
        theta = math.asin(s)
        angles.append(theta)
        c = math.cos(theta)
        x += lay.thickness_m * math.tan(theta)
        t += lay.thickness_m / (lay.velocity_mps * c)
    return (2 * x, 2 * t, angles)

def reflected_ray(layers: list[Layer], offset_m: float, iterations: int=80) -> RefractionResult:
    if not layers:
        return RefractionResult(0, 0, 0, ())
    p_lo = 0.0
    p_hi = 0.999 / max((l.velocity_mps for l in layers))
    for _ in range(iterations):
        p = (p_lo + p_hi) / 2
        x, _, _ = _offset_time(layers, p)
        if x < offset_m:
            p_lo = p
        else:
            p_hi = p
    p = (p_lo + p_hi) / 2
    x, t, a = _offset_time(layers, p)
    return RefractionResult(p, x, t, tuple(a))
