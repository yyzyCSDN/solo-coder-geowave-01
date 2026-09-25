from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class RefractionBranch:
    name: str
    intercept_s: float
    slowness_s_per_m: float

    def time(self, offset_m: float) -> float:
        return self.intercept_s + self.slowness_s_per_m * abs(offset_m)

def two_layer_branches(v1: float, v2: float, depth_m: float) -> list[RefractionBranch]:
    if not 0 < v1 < v2:
        raise ValueError('requires faster lower layer')
    critical = math.asin(v1 / v2)
    intercept = 2 * depth_m * math.cos(critical) / v1
    return [RefractionBranch('direct', 0, 1 / v1), RefractionBranch('head', intercept, 1 / v2)]

def crossover_distance(v1: float, v2: float, depth_m: float) -> float:
    a, b = two_layer_branches(v1, v2, depth_m)
    return b.intercept / (a.slowness_s_per_m - b.slowness_s_per_m)
