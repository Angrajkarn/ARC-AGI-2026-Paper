"""
Memory Store — persistent program library and experience replay.

Stores:
  - Successful programs (keyed by task type / feature hash)
  - Failed programs (to avoid re-exploring)
  - Primitive usage statistics
  - Transformation frequency counts
  - Search traces
  - Puzzle embeddings (feature vectors)

Backends:
  - in_memory : Dict (no persistence)
  - json      : JSON file on disk
  - sqlite    : SQLite database
"""

from __future__ import annotations

import json
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ProgramRecord:
    """A stored program with metadata."""

    program_json: str       # DSLProgram.to_json()
    task_id: str
    score: float
    train_accuracy: float
    is_successful: bool
    algorithm: str
    timestamp: float = field(default_factory=time.time)
    feature_hash: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class MemoryBackend(ABC):
    """Abstract backend for the memory store."""

    @abstractmethod
    def save(self, record: ProgramRecord) -> None: ...

    @abstractmethod
    def load_successful(self, limit: int = 100) -> List[ProgramRecord]: ...

    @abstractmethod
    def load_failed(self, limit: int = 100) -> List[ProgramRecord]: ...

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]: ...

    @abstractmethod
    def clear(self) -> None: ...


class InMemoryBackend(MemoryBackend):
    def __init__(self, max_size: int = 10000) -> None:
        self._records: List[ProgramRecord] = []
        self.max_size = max_size

    def save(self, record: ProgramRecord) -> None:
        if len(self._records) >= self.max_size:
            self._records.pop(0)
        self._records.append(record)

    def load_successful(self, limit: int = 100) -> List[ProgramRecord]:
        recs = [r for r in self._records if r.is_successful]
        return sorted(recs, key=lambda r: r.score, reverse=True)[:limit]

    def load_failed(self, limit: int = 100) -> List[ProgramRecord]:
        recs = [r for r in self._records if not r.is_successful]
        return recs[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        successful = [r for r in self._records if r.is_successful]
        return {
            "total_records": len(self._records),
            "successful": len(successful),
            "failed": len(self._records) - len(successful),
        }

    def clear(self) -> None:
        self._records.clear()


class JSONBackend(MemoryBackend):
    def __init__(self, filepath: str | Path, max_size: int = 10000) -> None:
        self._path = Path(filepath)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._records: List[Dict] = self._load_from_disk()

    def _load_from_disk(self) -> List[Dict]:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_to_disk(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._records[-self.max_size:], f, indent=2)

    def save(self, record: ProgramRecord) -> None:
        self._records.append(record.to_dict())
        self._save_to_disk()

    def load_successful(self, limit: int = 100) -> List[ProgramRecord]:
        recs = [r for r in self._records if r.get("is_successful")]
        recs.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return [ProgramRecord(**r) for r in recs[:limit]]

    def load_failed(self, limit: int = 100) -> List[ProgramRecord]:
        recs = [r for r in self._records if not r.get("is_successful")]
        return [ProgramRecord(**r) for r in recs[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        successful = sum(1 for r in self._records if r.get("is_successful"))
        return {
            "total_records": len(self._records),
            "successful": successful,
            "failed": len(self._records) - successful,
        }

    def clear(self) -> None:
        self._records.clear()
        if self._path.exists():
            self._path.unlink()


class MemoryStore:
    """High-level memory store interface.

    Usage::

        store = MemoryStore(backend="json", filepath="data/checkpoints/memory.json")
        store.save_success(program, task_id="abc", score=1.0, algorithm="beam")
        past_successes = store.recall_successful(limit=10)
    """

    def __init__(
        self,
        backend: str = "in_memory",
        filepath: Optional[str | Path] = None,
        max_size: int = 10000,
    ) -> None:
        if backend == "json":
            if filepath is None:
                filepath = "data/checkpoints/memory.json"
            self._backend: MemoryBackend = JSONBackend(filepath, max_size=max_size)
        elif backend == "in_memory":
            self._backend = InMemoryBackend(max_size=max_size)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        # Aggregate statistics
        self._primitive_counts: Dict[str, int] = {}
        self._rule_type_counts: Dict[str, int] = {}

    def save_success(
        self,
        program: Any,  # DSLProgram
        task_id: str,
        score: float,
        algorithm: str = "unknown",
        feature_hash: str = "",
    ) -> None:
        """Save a successful program."""
        record = ProgramRecord(
            program_json=program.to_json() if hasattr(program, "to_json") else str(program),
            task_id=task_id,
            score=score,
            train_accuracy=score,
            is_successful=True,
            algorithm=algorithm,
            feature_hash=feature_hash,
        )
        self._backend.save(record)
        self._update_primitive_counts(program)

    def save_failure(
        self,
        program: Any,
        task_id: str,
        score: float,
        algorithm: str = "unknown",
    ) -> None:
        """Record a failed program to avoid re-exploring."""
        record = ProgramRecord(
            program_json=program.to_json() if hasattr(program, "to_json") else str(program),
            task_id=task_id,
            score=score,
            train_accuracy=score,
            is_successful=False,
            algorithm=algorithm,
        )
        self._backend.save(record)

    def recall_successful(self, limit: int = 10) -> List[ProgramRecord]:
        """Return top successful program records."""
        return self._backend.load_successful(limit)

    def recall_failed_hashes(self) -> set:
        """Return a set of JSON hashes of known-failing programs."""
        failed = self._backend.load_failed(limit=500)
        return {r.program_json for r in failed}

    def primitive_statistics(self) -> Dict[str, int]:
        """Return primitive usage frequency counts."""
        return dict(sorted(self._primitive_counts.items(), key=lambda x: x[1], reverse=True))

    def get_stats(self) -> Dict[str, Any]:
        stats = self._backend.get_stats()
        stats["primitive_counts"] = self.primitive_statistics()
        return stats

    def clear(self) -> None:
        self._backend.clear()
        self._primitive_counts.clear()

    def _update_primitive_counts(self, program: Any) -> None:
        if hasattr(program, "instructions"):
            for instr in program.instructions:
                self._primitive_counts[instr.primitive] = (
                    self._primitive_counts.get(instr.primitive, 0) + 1
                )


__all__ = ["MemoryStore", "ProgramRecord"]
