from __future__ import annotations
from geowave.model.velocity import VelocityFunction, VelocityPick

def parse_velocity_functions(text: str) -> list[VelocityFunction]:
    groups = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split(',')
        if len(parts) < 5:
            raise ValueError(f'line {lineno}: expected id,x,y,time,velocity')
        ident = parts[0].strip()
        x, y, t, v = map(float, parts[1:5])
        q = float(parts[5]) if len(parts) > 5 else 1.0
        rec = groups.setdefault(ident, {'x': x, 'y': y, 'picks': []})
        if abs(rec['x'] - x) > 1e-06 or abs(rec['y'] - y) > 1e-06:
            raise ValueError(f'line {lineno}: coordinates changed')
        rec['picks'].append(VelocityPick(t, v, q))
    return [VelocityFunction(v['x'], v['y'], v['picks']) for v in groups.values()]

def dump_velocity_functions(funcs: list[VelocityFunction]) -> str:
    lines = []
    for i, f in enumerate(funcs):
        for p in f.picks:
            lines.append(f'V{i},{f.x:.3f},{f.y:.3f},{p.time_s:.6f},{p.velocity_mps:.3f},{p.confidence:.3f}')
    return '\n'.join(lines)
