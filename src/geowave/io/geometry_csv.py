from __future__ import annotations
import csv
from pathlib import Path
from geowave.model.traces import TraceHeader
REQUIRED = {'trace_id', 'shot_id', 'receiver_id', 'source_x', 'source_y', 'receiver_x', 'receiver_y', 'dt_s'}

def read_headers(path: str | Path) -> list[TraceHeader]:
    with Path(path).open(newline='', encoding='utf8') as f:
        r = csv.DictReader(f)
        if not REQUIRED.issubset(r.fieldnames or []):
            raise ValueError('missing geometry columns')
        out = []
        for row in r:
            kw = {k: row[k] for k in ['trace_id', 'shot_id', 'receiver_id']}
            for k in ['source_x', 'source_y', 'receiver_x', 'receiver_y', 'dt_s']:
                kw[k] = float(row[k])
            kw['start_s'] = float(row.get('start_s') or 0)
            kw['line'] = row.get('line') or ''
            kw['inline'] = int(row['inline']) if row.get('inline') else None
            kw['crossline'] = int(row['crossline']) if row.get('crossline') else None
            kw['elevation_m'] = float(row['elevation_m']) if row.get('elevation_m') else None
            out.append(TraceHeader(**kw))
    return out

def geometry_duplicates(headers: list[TraceHeader]) -> dict[str, list[int]]:
    seen = {}
    dups = {}
    for i, h in enumerate(headers):
        key = (round(h.source_x, 3), round(h.source_y, 3), round(h.receiver_x, 3), round(h.receiver_y, 3))
        if key in seen:
            dups.setdefault(str(key), [seen[key]]).append(i)
        else:
            seen[key] = i
    return dups
