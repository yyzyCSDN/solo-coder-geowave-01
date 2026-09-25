from __future__ import annotations
import math

def fresnel_radius(depth_m: float, velocity_mps: float, frequency_hz: float) -> float:
    if depth_m < 0 or velocity_mps <= 0 or frequency_hz <= 0:
        raise ValueError('invalid fresnel inputs')
    wavelength = velocity_mps / frequency_hz
    return math.sqrt(max(0, wavelength * depth_m / 2))

def migration_aperture(depth_m: float, opening_angle_deg: float) -> float:
    if not 0 < opening_angle_deg < 180:
        raise ValueError('opening angle')
    return depth_m * math.tan(math.radians(opening_angle_deg / 2))

def anti_alias_frequency(velocity_mps: float, dx_m: float, dip_deg: float) -> float:
    if dx_m <= 0:
        return float('inf')
    s = abs(math.sin(math.radians(dip_deg)))
    return velocity_mps / (2 * dx_m * max(s, 1e-06))
