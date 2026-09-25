from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from geowave.model.traces import TraceSet

@dataclass(frozen=True)
class IntegrityIssue:
    trace_id: str
    kind: str
    detail: str

def trace_sample_digest(samples: np.ndarray) -> str:
    data = np.asarray(samples, dtype='<f8').tobytes()
    return hashlib.sha256(data).hexdigest()

def inspect_integrity(dataset: TraceSet) -> list[IntegrityIssue]:
    issues: list[IntegrityIssue] = []
    seen_ids: set[str] = set()
    for trace in dataset.traces:
        trace_id = trace.header.trace_id
        if trace_id in seen_ids:
            issues.append(IntegrityIssue(trace_id, 'duplicate-id', 'trace identifier appears more than once'))
        seen_ids.add(trace_id)
        if not np.all(np.isfinite(trace.samples)):
            issues.append(IntegrityIssue(trace_id, 'non-finite', 'samples contain NaN or infinity'))
        if trace.samples.size == 0:
            issues.append(IntegrityIssue(trace_id, 'empty', 'trace has no samples'))
        if trace.header.dt_s <= 0:
            issues.append(IntegrityIssue(trace_id, 'sampling', 'sample interval is not positive'))
    return issues

def duplicate_content_groups(dataset: TraceSet) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for trace in dataset.traces:
        digest = trace_sample_digest(trace.samples)
        groups.setdefault(digest, []).append(trace.header.trace_id)
    return [ids for ids in groups.values() if len(ids) > 1]
