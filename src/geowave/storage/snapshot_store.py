from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from geowave.workflow.snapshot import AnalysisSnapshot

class SnapshotStore:

    def __init__(self, path: str | Path):
        self.path = str(path)
        with sqlite3.connect(self.path) as c:
            c.execute('CREATE TABLE IF NOT EXISTS snapshots(digest TEXT PRIMARY KEY,dataset_digest TEXT NOT NULL,recipe_digest TEXT NOT NULL,software_version TEXT NOT NULL,parameters TEXT NOT NULL,calibrations TEXT NOT NULL)')

    def put(self, snapshot: AnalysisSnapshot) -> str:
        digest = snapshot.fingerprint()
        with sqlite3.connect(self.path) as c:
            c.execute('INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?,?,?)', (digest, snapshot.dataset_digest, snapshot.recipe_digest, snapshot.software_version, json.dumps(snapshot.parameters, sort_keys=True), json.dumps(list(snapshot.calibration_refs))))
        return digest

    def get(self, digest: str) -> AnalysisSnapshot | None:
        with sqlite3.connect(self.path) as c:
            row = c.execute('SELECT dataset_digest,recipe_digest,software_version,parameters,calibrations FROM snapshots WHERE digest=?', (digest,)).fetchone()
        if row is None:
            return None
        return AnalysisSnapshot(row[0], row[1], row[2], json.loads(row[3]), tuple(json.loads(row[4])))

    def find_by_dataset(self, dataset_digest: str) -> list[str]:
        with sqlite3.connect(self.path) as c:
            rows = c.execute('SELECT digest FROM snapshots WHERE dataset_digest=? ORDER BY rowid', (dataset_digest,)).fetchall()
        return [r[0] for r in rows]
