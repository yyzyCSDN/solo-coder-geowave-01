from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json

@dataclass(frozen=True)
class DeliveryFile:
    path: str
    digest: str
    size_bytes: int
    role: str

@dataclass(frozen=True)
class DeliveryManifest:
    project: str
    run_id: str
    snapshot_digest: str
    files: tuple[DeliveryFile, ...]

    def canonical_bytes(self) -> bytes:
        payload = {'project': self.project, 'run_id': self.run_id, 'snapshot_digest': self.snapshot_digest, 'files': [asdict(f) for f in sorted(self.files, key=lambda x: x.path)]}
        return json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

def verify_manifest(manifest: DeliveryManifest, files: dict[str, bytes]) -> list[str]:
    issues = []
    for entry in manifest.files:
        data = files.get(entry.path)
        if data is None:
            issues.append(f'missing {entry.path}')
            continue
        if len(data) != entry.size_bytes:
            issues.append(f'size mismatch {entry.path}')
        if hashlib.sha256(data).hexdigest() != entry.digest:
            issues.append(f'digest mismatch {entry.path}')
    unexpected = sorted(set(files) - {f.path for f in manifest.files})
    issues.extend((f'unexpected {path}' for path in unexpected))
    return issues
