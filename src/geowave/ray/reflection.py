from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class ReflectionPoint:
    x: float
    z: float
    source_distance_m: float
    receiver_distance_m: float
    two_way_time_s: float

def flat_reflector_point(source_x: float, receiver_x: float, depth_m: float, velocity_mps: float) -> ReflectionPoint:
    if depth_m <= 0 or velocity_mps <= 0:
        raise ValueError('positive depth and velocity required')
    x = (source_x + receiver_x) / 2
    ds = math.hypot(x - source_x, depth_m)
    dr = math.hypot(receiver_x - x, depth_m)
    return ReflectionPoint(x, depth_m, ds, dr, (ds + dr) / velocity_mps)

def dipping_reflector_point(source_x: float, receiver_x: float, depth_at_zero_m: float, dip_deg: float, velocity_mps: float, iterations: int=80) -> ReflectionPoint:
    dip = math.radians(dip_deg)
    midpoint = (source_x + receiver_x) / 2
    span = max(abs(receiver_x - source_x), depth_at_zero_m) * 2
    lo = midpoint - span
    hi = midpoint + span

    def objective(x):
        z = depth_at_zero_m + x * math.tan(dip)
        if z <= 0:
            return float('inf')
        return math.hypot(x - source_x, z) + math.hypot(receiver_x - x, z)
    for _ in range(iterations):
        a = lo + (hi - lo) / 3
        b = hi - (hi - lo) / 3
        if objective(a) < objective(b):
            hi = b
        else:
            lo = a
    x = (lo + hi) / 2
    z = depth_at_zero_m + x * math.tan(dip)
    ds = math.hypot(x - source_x, z)
    dr = math.hypot(receiver_x - x, z)
    return ReflectionPoint(x, z, ds, dr, (ds + dr) / velocity_mps)
