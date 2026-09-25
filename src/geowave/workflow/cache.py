from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict
import threading

@dataclass(frozen=True)
class CacheEntry:
    key: str
    digest: str
    size_bytes: int

class ArtifactCache:

    def __init__(self, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError('max_bytes must be positive')
        self.max_bytes = max_bytes
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.used_bytes = 0
        self.lock = threading.RLock()

    def get(self, key: str) -> CacheEntry | None:
        with self.lock:
            entry = self.entries.get(key)
            if entry is not None:
                self.entries.move_to_end(key)
            return entry

    def put(self, entry: CacheEntry) -> list[CacheEntry]:
        evicted = []
        with self.lock:
            old = self.entries.pop(entry.key, None)
            if old is not None:
                self.used_bytes -= old.size_bytes
            self.entries[entry.key] = entry
            self.used_bytes += entry.size_bytes
            while self.used_bytes > self.max_bytes and self.entries:
                _, victim = self.entries.popitem(last=False)
                self.used_bytes -= victim.size_bytes
                evicted.append(victim)
        return evicted

    def invalidate_digest(self, digest: str) -> int:
        removed = 0
        with self.lock:
            for key in [k for k, v in self.entries.items() if v.digest == digest]:
                entry = self.entries.pop(key)
                self.used_bytes -= entry.size_bytes
                removed += 1
        return removed
