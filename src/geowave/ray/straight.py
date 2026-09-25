from __future__ import annotations
from dataclasses import dataclass
import math
from geowave.model.velocity import VelocityModel1D

@dataclass(frozen=True)
class RaySegment:
    x0: float
    z0: float
    x1: float
    z1: float
    velocity_mps: float

    @property
    def length_m(self) -> float:
        return math.hypot(self.x1 - self.x0, self.z1 - self.z0)

    @property
    def time_s(self) -> float:
        return self.length_m / self.velocity_mps

def straight_ray(x0: float, z0: float, x1: float, z1: float, velocity_mps: float) -> RaySegment:
    if velocity_mps <= 0:
        raise ValueError('velocity')
    return RaySegment(x0, z0, x1, z1, velocity_mps)

def sample_straight_time(model: VelocityModel1D, x0: float, z0: float, x1: float, z1: float, steps: int=512) -> float:
    if steps < 2:
        raise ValueError('steps')
    total = 0.0
    for i in range(steps - 1):
        a = i / (steps - 1)
        b = (i + 1) / (steps - 1)
        xa = x0 + (x1 - x0) * a
        za = z0 + (z1 - z0) * a
        xb = x0 + (x1 - x0) * b
        zb = z0 + (z1 - z0) * b
        v = model.velocity_at_depth((za + zb) / 2)
        total += math.hypot(xb - xa, zb - za) / v
    return total
