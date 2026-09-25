from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RunMetricChange:
    name: str
    before: float
    after: float
    relative_change: float

def compare_metrics(before: dict[str, float], after: dict[str, float]) -> list[RunMetricChange]:
    out = []
    for name in sorted(set(before) & set(after)):
        a = float(before[name])
        b = float(after[name])
        denom = abs(a) if abs(a) > 1e-12 else 1.0
        out.append(RunMetricChange(name, a, b, (b - a) / denom))
    return out

def regressions(changes: list[RunMetricChange], lower_is_better: set[str], threshold_fraction: float=0.05) -> list[RunMetricChange]:
    out = []
    for c in changes:
        if c.name in lower_is_better:
            if c.relative_change > threshold_fraction:
                out.append(c)
        elif c.relative_change < -threshold_fraction:
            out.append(c)
    return out

def markdown_diff(changes: list[RunMetricChange]) -> str:
    lines = ['| Metric | Before | After | Relative |', '|---|---:|---:|---:|']
    for c in changes:
        lines.append(f'| {c.name} | {c.before:.6g} | {c.after:.6g} | {c.relative_change:+.2%} |')
    return '\n'.join(lines)
