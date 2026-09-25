from __future__ import annotations
from dataclasses import dataclass
import json
import sqlite3
import time
from pathlib import Path

@dataclass(frozen=True)
class RunRecord:
    run_id: str
    dataset_id: str
    recipe_digest: str
    status: str
    output_digest: str | None
    metadata: dict
    updated_at: float

class RunStore:

    def __init__(self, path: str | Path):
        self.path = str(path)
        with sqlite3.connect(self.path) as c:
            c.execute('CREATE TABLE IF NOT EXISTS run_records(run_id TEXT PRIMARY KEY,dataset_id TEXT NOT NULL,recipe_digest TEXT NOT NULL,status TEXT NOT NULL,output_digest TEXT,metadata TEXT NOT NULL,updated REAL NOT NULL)')

    def put(self, record: RunRecord) -> None:
        with sqlite3.connect(self.path) as c:
            c.execute('INSERT INTO run_records VALUES(?,?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET status=excluded.status,output_digest=excluded.output_digest,metadata=excluded.metadata,updated=excluded.updated', (record.run_id, record.dataset_id, record.recipe_digest, record.status, record.output_digest, json.dumps(record.metadata, sort_keys=True), record.updated_at))

    def get(self, run_id: str) -> RunRecord | None:
        with sqlite3.connect(self.path) as c:
            row = c.execute('SELECT * FROM run_records WHERE run_id=?', (run_id,)).fetchone()
        if row is None:
            return None
        return RunRecord(row[0], row[1], row[2], row[3], row[4], json.loads(row[5]), float(row[6]))

    def list_by_status(self, status: str) -> list[RunRecord]:
        with sqlite3.connect(self.path) as c:
            rows = c.execute('SELECT * FROM run_records WHERE status=? ORDER BY updated', (status,)).fetchall()
        return [RunRecord(r[0], r[1], r[2], r[3], r[4], json.loads(r[5]), float(r[6])) for r in rows]
