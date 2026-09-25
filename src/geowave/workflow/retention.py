from __future__ import annotations
from dataclasses import dataclass
import time

@dataclass(frozen=True)
class RetentionItem:
    digest: str
    kind: str
    created_at: float
    referenced: bool
    size_bytes: int

@dataclass(frozen=True)
class RetentionDecision:
    digest: str
    action: str
    reason: str
    reclaim_bytes: int

def preview_retention(items: list[RetentionItem], max_age_s: float, now: float | None=None) -> list[RetentionDecision]:
    now = time.time() if now is None else now
    out = []
    for item in items:
        age = now - item.created_at
        if item.referenced:
            out.append(RetentionDecision(item.digest, 'keep', 'still referenced', 0))
        elif age < max_age_s:
            out.append(RetentionDecision(item.digest, 'keep', 'younger than retention window', 0))
        else:
            out.append(RetentionDecision(item.digest, 'delete', 'unreferenced and expired', item.size_bytes))
    return out

def reclaimable_bytes(decisions: list[RetentionDecision]) -> int:
    return sum((d.reclaim_bytes for d in decisions if d.action == 'delete'))
