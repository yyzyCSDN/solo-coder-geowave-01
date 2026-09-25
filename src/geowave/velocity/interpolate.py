from __future__ import annotations
import math
from geowave.model.velocity import VelocityFunction, VelocityPick

def idw_velocity(functions: list[VelocityFunction], x: float, y: float, times: list[float], power: float=2.0, max_radius: float | None=None) -> VelocityFunction:
    if not functions:
        raise ValueError('no functions')
    picks = []
    for t in times:
        num = den = 0.0
        exact = None
        for f in functions:
            d = math.hypot(f.x - x, f.y - y)
            if d < 1e-09:
                exact = f.at(t)
                break
            if max_radius is not None and d > max_radius:
                continue
            w = 1 / d ** power
            num += w * f.at(t)
            den += w
        if exact is not None:
            v = exact
        elif den:
            v = num / den
        else:
            v = min(functions, key=lambda f: math.hypot(f.x - x, f.y - y)).at(t)
        picks.append(VelocityPick(t, v))
    return VelocityFunction(x, y, picks)

def smooth_time(func: VelocityFunction, radius: int=1) -> VelocityFunction:
    ps = func.picks
    out = []
    for i, p in enumerate(ps):
        a = max(0, i - radius)
        b = min(len(ps), i + radius + 1)
        v = sum((q.velocity_mps * q.confidence for q in ps[a:b])) / sum((q.confidence for q in ps[a:b]))
        out.append(VelocityPick(p.time_s, v, p.confidence))
    return VelocityFunction(func.x, func.y, out)
