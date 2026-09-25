from __future__ import annotations
from dataclasses import dataclass
import threading
from geowave.model.traces import TraceSet
from geowave.workflow.executor import dataset_digest

@dataclass(frozen=True)
class DatasetRevision:
    dataset_id: str
    revision: int
    digest: str
    dataset: TraceSet
    parent_digest: str | None

class DatasetRegistry:

    def __init__(self):
        self._items: dict[str, list[DatasetRevision]] = {}
        self._lock = threading.RLock()

    def add(self, dataset_id: str, dataset: TraceSet) -> DatasetRevision:
        digest = dataset_digest(dataset)
        with self._lock:
            history = self._items.setdefault(dataset_id, [])
            if history and history[-1].digest == digest:
                return history[-1]
            revision = DatasetRevision(dataset_id, len(history) + 1, digest, dataset, history[-1].digest if history else None)
            history.append(revision)
            return revision

    def latest(self, dataset_id: str) -> DatasetRevision | None:
        with self._lock:
            values = self._items.get(dataset_id, [])
            return values[-1] if values else None

    def by_digest(self, dataset_id: str, digest: str) -> DatasetRevision | None:
        with self._lock:
            for revision in self._items.get(dataset_id, []):
                if revision.digest == digest:
                    return revision
        return None

    def lineage(self, dataset_id: str) -> list[tuple[int, str, str | None]]:
        with self._lock:
            return [(r.revision, r.digest, r.parent_digest) for r in self._items.get(dataset_id, [])]
