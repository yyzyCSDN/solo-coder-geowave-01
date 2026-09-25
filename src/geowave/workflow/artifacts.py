from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, os

@dataclass(frozen=True)
class Artifact:
    digest: str
    media_type: str
    size: int
    path: str
    metadata: dict

class ArtifactStore:

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, data: bytes, media_type: str, metadata: dict | None=None) -> Artifact:
        digest = hashlib.sha256(data).hexdigest()
        p = self.root / digest[:2] / digest
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            tmp = p.with_suffix('.tmp')
            tmp.write_bytes(data)
            os.replace(tmp, p)
        meta = dict(metadata or {})
        p.with_suffix('.json').write_text(json.dumps({'media_type': media_type, 'metadata': meta}, sort_keys=True), encoding='utf8')
        return Artifact(digest, media_type, len(data), str(p), meta)

    def get(self, digest: str) -> bytes:
        p = self.root / digest[:2] / digest
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('artifact digest mismatch')
        return data

    def exists(self, digest: str) -> bool:
        return (self.root / digest[:2] / digest).exists()
