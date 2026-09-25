from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class HorizonPick:
    x: float
    y: float
    time_s: float
    quality: float = 1.0

@dataclass
class Horizon:
    name: str
    picks: list[HorizonPick]

    def nearest(self, x: float, y: float) -> HorizonPick | None:
        if not self.picks:
            return None
        return min(self.picks, key=lambda p: (p.x - x) ** 2 + (p.y - y) ** 2)

    def bounding_box(self) -> tuple[float, float, float, float]:
        if not self.picks:
            return (0, 0, 0, 0)
        xs = [p.x for p in self.picks]
        ys = [p.y for p in self.picks]
        return (min(xs), max(xs), min(ys), max(ys))

    def robust_time_range(self) -> tuple[float, float]:
        if not self.picks:
            return (0, 0)
        a = np.asarray([p.time_s for p in self.picks])
        return (float(np.percentile(a, 5)), float(np.percentile(a, 95)))
