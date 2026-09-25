from __future__ import annotations
from dataclasses import dataclass
from geowave.model.velocity import VelocityFunction, VelocityPick

@dataclass(frozen=True)
class VelocityScenario:
    name: str
    scale: float
    time_shift_s: float = 0.0

def apply_scenario(func: VelocityFunction, scenario: VelocityScenario) -> VelocityFunction:
    if scenario.scale <= 0:
        raise ValueError('scenario scale must be positive')
    picks = [VelocityPick(max(0.0, p.time_s + scenario.time_shift_s), p.velocity_mps * scenario.scale, p.confidence) for p in func.picks]
    return VelocityFunction(func.x, func.y, picks)

def scenario_grid(base: VelocityFunction, scales: list[float], shifts_s: list[float]) -> list[tuple[VelocityScenario, VelocityFunction]]:
    out = []
    for scale in scales:
        for shift in shifts_s:
            scenario = VelocityScenario(f'scale={scale:g},shift={shift:g}', scale, shift)
            out.append((scenario, apply_scenario(base, scenario)))
    return out

def bracket_velocity(base: VelocityFunction, fraction: float=0.05) -> tuple[VelocityFunction, VelocityFunction]:
    return (apply_scenario(base, VelocityScenario('low', 1 - fraction)), apply_scenario(base, VelocityScenario('high', 1 + fraction)))
