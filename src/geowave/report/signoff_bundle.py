from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
from geowave.qc.signoff import SignoffResult

@dataclass(frozen=True)
class SignoffBundle:
    dataset_digest: str
    recipe_digest: str
    snapshot_digest: str
    signoff: SignoffResult
    report_digest: str

    def canonical_bytes(self) -> bytes:
        payload = {'dataset_digest': self.dataset_digest, 'recipe_digest': self.recipe_digest, 'snapshot_digest': self.snapshot_digest, 'signoff': {'passed': self.signoff.passed, 'checks': [asdict(c) for c in self.signoff.checks]}, 'report_digest': self.report_digest}
        return json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

def build_signoff_bundle(dataset_digest: str, recipe_digest: str, snapshot_digest: str, signoff: SignoffResult, report_bytes: bytes) -> SignoffBundle:
    return SignoffBundle(dataset_digest, recipe_digest, snapshot_digest, signoff, hashlib.sha256(report_bytes).hexdigest())

def compare_signoff_bundles(a: SignoffBundle, b: SignoffBundle) -> list[str]:
    changes = []
    if a.dataset_digest != b.dataset_digest:
        changes.append('dataset changed')
    if a.recipe_digest != b.recipe_digest:
        changes.append('recipe changed')
    if a.snapshot_digest != b.snapshot_digest:
        changes.append('snapshot changed')
    if a.signoff.passed != b.signoff.passed:
        changes.append('signoff state changed')
    if a.report_digest != b.report_digest:
        changes.append('report changed')
    return changes
