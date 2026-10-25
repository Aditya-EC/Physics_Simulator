"""Persistent, unique object identifiers (OBJ-000001, ...).

IDs are never reused inside a project: the allocator's counter only
increases and is saved with the project. Renaming or editing an object never
changes its ID.
"""
from __future__ import annotations

import re

ID_PATTERN = re.compile(r"^OBJ-(\d{6,})$")


class DuplicateIdError(ValueError):
    pass


class IdAllocator:
    prefix = "OBJ"

    def __init__(self, next_value: int = 1) -> None:
        self._next = max(1, int(next_value))
        self._used: set[str] = set()

    @property
    def next_value(self) -> int:
        return self._next

    def allocate(self) -> str:
        while True:
            oid = f"{self.prefix}-{self._next:06d}"
            self._next += 1
            if oid not in self._used:
                self._used.add(oid)
                return oid

    def reserve(self, oid: str) -> None:
        """Register an existing ID (e.g. when loading a project)."""
        m = ID_PATTERN.match(oid)
        if not m:
            raise ValueError(f"Invalid object ID '{oid}'")
        if oid in self._used:
            raise DuplicateIdError(f"Duplicate object ID '{oid}'")
        self._used.add(oid)
        self._next = max(self._next, int(m.group(1)) + 1)

    def release(self, oid: str) -> None:
        """Forget that an ID is in use (the counter is NOT decremented)."""
        self._used.discard(oid)

    def is_used(self, oid: str) -> bool:
        return oid in self._used

    def reset(self, next_value: int = 1) -> None:
        self._next = max(1, int(next_value))
        self._used.clear()
