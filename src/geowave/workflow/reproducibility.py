from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from geowave.workflow.snapshot import AnalysisSnapshot

@dataclass(frozen=True)
class EnvironmentFingerprint:
    python_version: str
    platform: str
    dependencies: tuple[tuple[str, str], ...]

    def digest(self) -> str:
        raw = json.dumps({'python_version': self.python_version, 'platform': self.platform, 'dependencies': list(self.dependencies)}, sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(raw).hexdigest()

def reproducibility_key(snapshot: AnalysisSnapshot, environment: EnvironmentFingerprint) -> str:
    return hashlib.sha256((snapshot.fingerprint() + environment.digest()).encode()).hexdigest()

def explain_reproducibility(a: tuple[AnalysisSnapshot, EnvironmentFingerprint], b: tuple[AnalysisSnapshot, EnvironmentFingerprint]) -> list[str]:
    reasons = []
    if a[0].dataset_digest != b[0].dataset_digest:
        reasons.append('dataset digest changed')
    if a[0].recipe_digest != b[0].recipe_digest:
        reasons.append('recipe changed')
    if a[0].parameters != b[0].parameters:
        reasons.append('analysis parameters changed')
    if a[1].digest() != b[1].digest():
        reasons.append('runtime environment changed')
    return reasons
