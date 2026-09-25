from __future__ import annotations
from dataclasses import dataclass
import math
from geowave.model.horizon import HorizonPick

@dataclass(frozen=True)
class FaultSegment:
    x0: float
    y0: float
    x1: float
    y1: float

    def side(self, x: float, y: float) -> float:
        return (self.x1 - self.x0) * (y - self.y0) - (self.y1 - self.y0) * (x - self.x0)

def estimate_throw(fault: FaultSegment, picks: list[HorizonPick], search_m: float) -> float | None:
    left = []
    right = []
    vx = fault.x1 - fault.x0
    vy = fault.y1 - fault.y0
    norm = math.hypot(vx, vy)
    if norm == 0:
        return None
    for p in picks:
        d = abs(fault.side(p.x, p.y)) / norm
        if d <= search_m:
            (left if fault.side(p.x, p.y) >= 0 else right).append(p.time_s)
    if not left or not right:
        return None
    return sum(right) / len(right) - sum(left) / len(left)

def segment_intersection(a: FaultSegment, b: FaultSegment) -> tuple[float, float] | None:
    x1, y1, x2, y2 = (a.x0, a.y0, a.x1, a.y1)
    x3, y3, x4, y4 = (b.x0, b.y0, b.x1, b.y1)
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (px, py)
