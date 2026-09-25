from __future__ import annotations
import sqlite3, json, time, hashlib
from pathlib import Path
SCHEMA = '\nCREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,name TEXT NOT NULL,digest TEXT NOT NULL,metadata TEXT NOT NULL,created REAL NOT NULL);\nCREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,dataset_id TEXT NOT NULL,recipe_digest TEXT NOT NULL,status TEXT NOT NULL,result_digest TEXT,created REAL NOT NULL,updated REAL NOT NULL);\nCREATE TABLE IF NOT EXISTS audit(seq INTEGER PRIMARY KEY AUTOINCREMENT,kind TEXT NOT NULL,payload TEXT NOT NULL,prev_hash TEXT NOT NULL,entry_hash TEXT NOT NULL,created REAL NOT NULL);\n'

class Database:

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.initialize()

    def connect(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        c.execute('PRAGMA foreign_keys=ON')
        return c

    def initialize(self):
        with self.connect() as c:
            c.executescript(SCHEMA)

    def put_dataset(self, id: str, name: str, digest: str, metadata: dict):
        with self.connect() as c:
            c.execute('INSERT OR REPLACE INTO datasets VALUES(?,?,?,?,?)', (id, name, digest, json.dumps(metadata, sort_keys=True), time.time()))

    def get_dataset(self, id: str):
        with self.connect() as c:
            r = c.execute('SELECT * FROM datasets WHERE id=?', (id,)).fetchone()
            return dict(r) if r else None

    def put_run(self, id: str, dataset_id: str, recipe_digest: str, status: str, result_digest: str | None=None):
        now = time.time()
        with self.connect() as c:
            c.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,result_digest=excluded.result_digest,updated=excluded.updated', (id, dataset_id, recipe_digest, status, result_digest, now, now))

    def append_audit(self, kind: str, payload: dict) -> str:
        body = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        now = time.time()
        with self.connect() as c:
            row = c.execute('SELECT entry_hash FROM audit ORDER BY seq DESC LIMIT 1').fetchone()
            prev = row[0] if row else '0' * 64
            entry = hashlib.sha256((prev + kind + body).encode()).hexdigest()
            c.execute('INSERT INTO audit(kind,payload,prev_hash,entry_hash,created) VALUES(?,?,?,?,?)', (kind, body, prev, entry, now))
            return entry

    def audit_valid(self) -> bool:
        prev = '0' * 64
        with self.connect() as c:
            rows = c.execute('SELECT kind,payload,prev_hash,entry_hash FROM audit ORDER BY seq').fetchall()
        for r in rows:
            expected = hashlib.sha256((prev + r['kind'] + r['payload']).encode()).hexdigest()
            if r['prev_hash'] != prev or r['entry_hash'] != expected:
                return False
            prev = r['entry_hash']
        return True
