from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, json, time

@dataclass(frozen=True)
class ProvenanceRecord:
    run_id: str
    step_index: int
    operation: str
    input_digest: str
    output_digest: str
    parameters: dict
    started_at: float
    finished_at: float

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

class ProvenanceChain:

    def __init__(self):
        self.records = []

    def append(self, record: ProvenanceRecord) -> str:
        self.records.append(record)
        return record.digest()

    def root(self) -> str:
        h = hashlib.sha256()
        for r in self.records:
            h.update(r.digest().encode())
        return h.hexdigest()

    def verify(self) -> bool:
        return self.root() == ProvenanceChain.from_json(self.to_json()).root()

    def to_json(self) -> str:
        return json.dumps([asdict(r) for r in self.records], sort_keys=True)

    @classmethod
    def from_json(cls, text: str):
        c = cls()
        c.records = [ProvenanceRecord(**r) for r in json.loads(text)]
        return c
