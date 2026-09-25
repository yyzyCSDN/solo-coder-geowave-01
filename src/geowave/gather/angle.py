from __future__ import annotations
from dataclasses import dataclass
import math
from .cmp import Gather

@dataclass(frozen=True)
class AngleTrace:
    trace_id: str
    angle_deg: float
    offset_m: float

def approximate_angles(gather: Gather, reflection_time_s: float, velocity_mps: float) -> list[AngleTrace]:
    if reflection_time_s <= 0 or velocity_mps <= 0:
        raise ValueError('positive time and velocity')
    depth = 0.5 * reflection_time_s * velocity_mps
    out = []
    for tr in gather.traces:
        half = tr.header.offset_m / 2
        angle = math.degrees(math.atan2(half, depth))
        out.append(AngleTrace(tr.header.trace_id, angle, tr.header.offset_m))
    return out

def angle_bins(angles: list[AngleTrace], edges_deg: list[float]) -> dict[tuple[float, float], list[str]]:
    out = {(a, b): [] for a, b in zip(edges_deg, edges_deg[1:])}
    for item in angles:
        for a, b in out:
            if a <= item.angle_deg < b:
                out[a, b].append(item.trace_id)
                break
    return out
