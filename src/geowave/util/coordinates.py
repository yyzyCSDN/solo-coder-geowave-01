from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class LocalFrame:
    origin_x: float
    origin_y: float
    azimuth_deg: float

    def to_local(self, x: float, y: float) -> tuple[float, float]:
        dx = x - self.origin_x
        dy = y - self.origin_y
        a = math.radians(self.azimuth_deg)
        inline = dx * math.sin(a) + dy * math.cos(a)
        crossline = dx * math.cos(a) - dy * math.sin(a)
        return (inline, crossline)

    def to_world(self, inline: float, crossline: float) -> tuple[float, float]:
        a = math.radians(self.azimuth_deg)
        dx = inline * math.sin(a) + crossline * math.cos(a)
        dy = inline * math.cos(a) - crossline * math.sin(a)
        return (self.origin_x + dx, self.origin_y + dy)

def azimuth_deg(x0: float, y0: float, x1: float, y1: float) -> float:
    return (math.degrees(math.atan2(x1 - x0, y1 - y0)) + 360.0) % 360.0
