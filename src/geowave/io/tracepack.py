from __future__ import annotations
import json, struct, hashlib
from pathlib import Path
import numpy as np
from geowave.model.traces import Trace, TraceHeader, TraceSet
MAGIC = b'GWTP1\x00'

def write_tracepack(path: str | Path, dataset: TraceSet) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open('wb') as f:
        f.write(MAGIC)
        h.update(MAGIC)
        meta = {'name': dataset.name, 'count': len(dataset.traces)}
        raw = json.dumps(meta, separators=(',', ':')).encode()
        f.write(struct.pack('>I', len(raw)))
        f.write(raw)
        h.update(raw)
        for tr in dataset.traces:
            hd = tr.header.__dict__ | {'tags': tr.tags, 'n': int(tr.samples.size)}
            hb = json.dumps(hd, separators=(',', ':')).encode()
            f.write(struct.pack('>I', len(hb)))
            f.write(hb)
            samples = np.asarray(tr.samples, dtype='>f4').tobytes()
            f.write(samples)
            h.update(hb)
            h.update(samples)
    return h.hexdigest()

def read_tracepack(path: str | Path) -> TraceSet:
    p = Path(path)
    with p.open('rb') as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError('invalid tracepack magic')
        n = struct.unpack('>I', f.read(4))[0]
        meta = json.loads(f.read(n))
        traces = []
        for _ in range(meta['count']):
            m = struct.unpack('>I', f.read(4))[0]
            hd = json.loads(f.read(m))
            count = int(hd.pop('n'))
            tags = hd.pop('tags', {})
            data = f.read(count * 4)
            if len(data) != count * 4:
                raise ValueError('truncated trace samples')
            traces.append(Trace(TraceHeader(**hd), np.frombuffer(data, dtype='>f4').astype(float), tags))
    return TraceSet(traces, meta.get('name', 'dataset'))
