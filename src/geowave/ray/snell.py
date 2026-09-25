from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class SnellResult:
    incident_deg: float
    transmitted_deg: float | None
    critical: bool
    ray_parameter_s_per_m: float

def transmit_angle(v1: float, v2: float, incident_deg: float) -> SnellResult:
    if v1 <= 0 or v2 <= 0:
        raise ValueError('velocities must be positive')
    theta1 = math.radians(incident_deg)
    p = math.sin(theta1) / v1
    s = p * v2
    if abs(s) > 1:
        return SnellResult(incident_deg, None, True, p)
    return SnellResult(incident_deg, math.degrees(math.asin(s)), False, p)

def critical_angle_deg(v1: float, v2: float) -> float | None:
    if v2 <= v1:
        return None
    return math.degrees(math.asin(v1 / v2))

def ray_parameter_from_takeoff(velocity_mps: float, angle_deg: float) -> float:
    return math.sin(math.radians(angle_deg)) / velocity_mps
