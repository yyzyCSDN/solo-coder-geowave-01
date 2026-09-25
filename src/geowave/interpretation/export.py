from __future__ import annotations
import csv
import io
import json
from geowave.model.horizon import Horizon

def horizon_csv(horizon: Horizon) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(['horizon', 'x', 'y', 'time_s', 'quality'])
    for p in horizon.picks:
        writer.writerow([horizon.name, f'{p.x:.6f}', f'{p.y:.6f}', f'{p.time_s:.9f}', f'{p.quality:.6f}'])
    return stream.getvalue()

def horizon_geojson(horizon: Horizon) -> str:
    ordered = sorted(horizon.picks, key=lambda p: (p.x, p.y))
    coordinates = [[p.x, p.y, p.time_s] for p in ordered]
    feature = {'type': 'Feature', 'properties': {'name': horizon.name, 'count': len(ordered)}, 'geometry': {'type': 'LineString', 'coordinates': coordinates}}
    return json.dumps({'type': 'FeatureCollection', 'features': [feature]}, sort_keys=True)

def horizon_digest_payload(horizon: Horizon) -> bytes:
    rows = [{'x': p.x, 'y': p.y, 'time_s': p.time_s, 'quality': p.quality} for p in sorted(horizon.picks, key=lambda p: (p.x, p.y, p.time_s))]
    return json.dumps({'name': horizon.name, 'picks': rows}, sort_keys=True, separators=(',', ':')).encode()
