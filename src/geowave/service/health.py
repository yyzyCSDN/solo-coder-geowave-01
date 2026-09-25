from __future__ import annotations
from dataclasses import dataclass
import os
import shutil
from geowave.service.state import ServiceState

@dataclass(frozen=True)
class HealthCheck:
    name: str
    ok: bool
    detail: str

def health_checks(state: ServiceState) -> list[HealthCheck]:
    checks = []
    try:
        valid = state.db.audit_valid()
        checks.append(HealthCheck('audit', valid, 'valid' if valid else 'hash chain invalid'))
    except Exception as exc:
        checks.append(HealthCheck('audit', False, str(exc)))
    root = state.artifacts.root
    usage = shutil.disk_usage(root)
    checks.append(HealthCheck('artifact-space', usage.free > 10 * 1024 * 1024, f'free={usage.free}'))
    checks.append(HealthCheck('operations', len(state.executor.operations) >= 10, f'registered={len(state.executor.operations)}'))
    checks.append(HealthCheck('datasets', True, f'loaded={len(state.datasets)}'))
    return checks

def health_summary(state: ServiceState) -> dict[str, object]:
    checks = health_checks(state)
    return {'ready': all((c.ok for c in checks)), 'checks': [c.__dict__ for c in checks]}
