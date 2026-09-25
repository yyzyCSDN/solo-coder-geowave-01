from __future__ import annotations
from dataclasses import dataclass
import hashlib

@dataclass(frozen=True)
class ChainEntry:
    sequence: int
    payload_digest: str
    previous_hash: str
    entry_hash: str

def append_entry(entries: list[ChainEntry], payload: bytes) -> ChainEntry:
    previous = entries[-1].entry_hash if entries else '0' * 64
    payload_digest = hashlib.sha256(payload).hexdigest()
    sequence = len(entries) + 1
    entry_hash = hashlib.sha256(f'{sequence}:{previous}:{payload_digest}'.encode()).hexdigest()
    return ChainEntry(sequence, payload_digest, previous, entry_hash)

def verify_chain(entries: list[ChainEntry]) -> bool:
    previous = '0' * 64
    for expected_sequence, entry in enumerate(entries, 1):
        if entry.sequence != expected_sequence:
            return False
        if entry.previous_hash != previous:
            return False
        expected = hashlib.sha256(f'{entry.sequence}:{entry.previous_hash}:{entry.payload_digest}'.encode()).hexdigest()
        if entry.entry_hash != expected:
            return False
        previous = entry.entry_hash
    return True

def chain_root(entries: list[ChainEntry]) -> str:
    return entries[-1].entry_hash if entries else '0' * 64
