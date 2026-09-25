from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import hashlib
import json
import time
from geowave.model.traces import TraceSet
from geowave.workflow.recipe import Recipe
from geowave.workflow.provenance import ProvenanceChain, ProvenanceRecord
Operation = Callable[[TraceSet, dict[str, object]], TraceSet]

@dataclass(frozen=True)
class RunResult:
    dataset: TraceSet
    provenance: ProvenanceChain
    input_digest: str
    output_digest: str

def dataset_digest(dataset: TraceSet) -> str:
    h = hashlib.sha256()
    h.update(dataset.name.encode())
    for tr in sorted(dataset.traces, key=lambda t: t.header.trace_id):
        h.update(tr.header.trace_id.encode())
        h.update(json.dumps(tr.header.__dict__, sort_keys=True, default=str).encode())
        h.update(tr.samples.astype('<f8').tobytes())
    return h.hexdigest()

class Executor:

    def __init__(self):
        self.operations: dict[str, Operation] = {}

    def register(self, name: str, operation: Operation) -> None:
        if name in self.operations:
            raise ValueError(f'operation already registered: {name}')
        self.operations[name] = operation

    def execute(self, dataset: TraceSet, recipe: Recipe, run_id: str) -> RunResult:
        current = dataset
        chain = ProvenanceChain()
        original = dataset_digest(current)
        for index, step in enumerate(recipe.steps):
            if step.operation not in self.operations:
                raise KeyError(step.operation)
            before = dataset_digest(current)
            started = time.time()
            current = self.operations[step.operation](current, dict(step.parameters))
            finished = time.time()
            after = dataset_digest(current)
            chain.append(ProvenanceRecord(run_id=run_id, step_index=index, operation=step.operation, input_digest=before, output_digest=after, parameters=dict(step.parameters), started_at=started, finished_at=finished))
        return RunResult(current, chain, original, dataset_digest(current))
