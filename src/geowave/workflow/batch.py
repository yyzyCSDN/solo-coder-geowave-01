from __future__ import annotations
from dataclasses import dataclass, field
import time
import uuid
from geowave.workflow.executor import Executor, RunResult
from geowave.workflow.recipe import Recipe
from geowave.model.traces import TraceSet

@dataclass
class BatchItem:
    dataset_id: str
    status: str = 'pending'
    output_digest: str | None = None
    error: str | None = None

@dataclass
class BatchRun:
    batch_id: str
    items: list[BatchItem]
    created_at: float = field(default_factory=time.time)

def run_batch(executor: Executor, datasets: dict[str, TraceSet], recipe: Recipe, stop_on_error: bool=False) -> BatchRun:
    batch = BatchRun(uuid.uuid4().hex, [BatchItem(k) for k in sorted(datasets)])
    for item in batch.items:
        try:
            item.status = 'running'
            result: RunResult = executor.execute(datasets[item.dataset_id], recipe, batch.batch_id + ':' + item.dataset_id)
            item.output_digest = result.output_digest
            item.status = 'complete'
        except Exception as exc:
            item.status = 'failed'
            item.error = f'{type(exc).__name__}: {exc}'
            if stop_on_error:
                break
    return batch

def batch_summary(batch: BatchRun) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in batch.items:
        counts[item.status] = counts.get(item.status, 0) + 1
    return counts
