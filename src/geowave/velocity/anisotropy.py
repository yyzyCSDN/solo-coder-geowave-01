from __future__ import annotations
import math

def nonhyperbolic_moveout(t0_s: float, offset_m: float, vnmo_mps: float, eta: float) -> float:
    if t0_s <= 0 or vnmo_mps <= 0:
        return 0.0
    x2 = offset_m * offset_m
    v2 = vnmo_mps * vnmo_mps
    base = t0_s * t0_s + x2 / v2
    correction = 2 * eta * x2 * x2 / (v2 * v2 * t0_s * t0_s + (1 + 2 * eta) * v2 * x2 + 1e-12)
    return math.sqrt(max(0.0, base - correction))

def thomsen_phase_velocity(vp0: float, epsilon: float, delta: float, theta_rad: float) -> float:
    s2 = math.sin(theta_rad) ** 2
    c2 = math.cos(theta_rad) ** 2
    return vp0 * math.sqrt(max(1e-12, 1 + 2 * delta * s2 * c2 + 2 * epsilon * s2 * s2))

def eta_from_thomsen(epsilon: float, delta: float) -> float:
    return (epsilon - delta) / (1 + 2 * delta)
