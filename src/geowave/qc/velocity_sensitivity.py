from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.gather.cmp import Gather
from geowave.gather.nmo import nmo_correct
from geowave.gather.stack import linear_stack
from geowave.model.velocity import VelocityFunction
from geowave.velocity.scenario import apply_scenario, VelocityScenario

@dataclass(frozen=True)
class VelocitySensitivity:
    scale: float
    stack_rms: float
    stack_correlation: float

def stack_for_velocity(gather: Gather, velocity: VelocityFunction) -> np.ndarray:
    corrected = Gather(gather.key, [nmo_correct(t, velocity) for t in gather.traces], gather.x, gather.y)
    return linear_stack(corrected).samples

def sensitivity_scan(gather: Gather, base: VelocityFunction, scales: list[float]) -> list[VelocitySensitivity]:
    reference = stack_for_velocity(gather, base)
    result = []
    for scale in scales:
        candidate = apply_scenario(base, VelocityScenario(str(scale), scale))
        stack = stack_for_velocity(gather, candidate)
        n = min(len(reference), len(stack))
        a = reference[:n]
        b = stack[:n]
        corr = float(np.corrcoef(a, b)[0, 1]) if n > 1 and np.std(a) and np.std(b) else 0.0
        result.append(VelocitySensitivity(scale, float(np.sqrt(np.mean(b * b))), corr))
    return result
