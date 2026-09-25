from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json

@dataclass(frozen=True)
class EvidenceEntry:
    name: str
    digest: str
    media_type: str
    role: str

@dataclass(frozen=True)
class EvidenceBundle:
    run_id: str
    snapshot_digest: str
    entries: tuple[EvidenceEntry, ...]

    def manifest_bytes(self) -> bytes:
        payload = {'run_id': self.run_id, 'snapshot_digest': self.snapshot_digest, 'entries': [asdict(e) for e in sorted(self.entries, key=lambda x: x.name)]}
        return json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()

    def digest(self) -> str:
        return hashlib.sha256(self.manifest_bytes()).hexdigest()

def verify_bundle(bundle: EvidenceBundle, available: dict[str, bytes]) -> list[str]:
    issues = []
    for entry in bundle.entries:
        data = available.get(entry.name)
        if data is None:
            issues.append(f'missing {entry.name}')
            continue
        actual = hashlib.sha256(data).hexdigest()
        if actual != entry.digest:
            issues.append(f'digest mismatch {entry.name}')
    return issues
