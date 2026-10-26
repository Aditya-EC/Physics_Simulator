"""Thread-safe store of sampled simulation data.

The simulation engine appends one *sample* (one row) every N steps from its
worker thread; the GUI reads copies of columns at any time. Rows are stored in
one 2-D float64 buffer whose columns are named by keys such as
``("OBJ-000001", "origin")`` (3 columns), so every column shares the same
sample times.

Memory is bounded by ``max_bytes``. When the buffer is full the overflow
policy applies:

* ``decimate`` – drop every second stored sample and double the sampling
  stride from then on: the whole run is kept at half the resolution.
* ``window``   – drop the oldest quarter of the samples (recent history).
* ``stop``     – stop recording (an event is logged).
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

Key = tuple
OVERFLOW_POLICIES = ("decimate", "window", "stop")


@dataclass
class Event:
    time: float          # simulated time
    step: int
    kind: str            # start | pause | resume | stop | reset | breakpoint | change | warning | error | recording
    message: str
    wall: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"time": self.time, "step": self.step, "kind": self.kind, "message": self.message,
                "wall_time": self.wall}


class Recording:
    def __init__(self, max_bytes: int = 256 * 2 ** 20, overflow: str = "decimate", interval_steps: int = 1) -> None:
        if overflow not in OVERFLOW_POLICIES:
            raise ValueError(f"Unknown overflow policy '{overflow}'")
        self._lock = threading.RLock()
        self.max_bytes = int(max_bytes)
        self.overflow = overflow
        self.interval_steps = max(1, int(interval_steps))
        self.stride = 1                  # extra factor after decimation
        self.columns: dict[Key, slice] = {"t": slice(0, 1), "step": slice(1, 2)}
        self.width = 2
        self._buf = np.empty((0, self.width))
        self.n = 0
        self.version = 0                 # increments on every change (for cheap cache checks)
        self.decimations = 0
        self.dropped = 0                 # samples discarded by the window policy
        self.full_stop = False
        self.events: list[Event] = []
        self.meta: dict = {"objects": {}, "processes": {}, "settings": {}, "started": time.time()}

    # ---- schema ------------------------------------------------------------------------------------------
    def ensure_columns(self, specs: list[tuple[Key, int]]) -> None:
        """Add columns (key, width) that do not exist yet; earlier samples get NaN."""
        with self._lock:
            new = [(k, w) for k, w in specs if k not in self.columns]
            if not new:
                return
            for k, w in new:
                self.columns[k] = slice(self.width, self.width + w)
                self.width += w
            buf = np.full((self._buf.shape[0], self.width), np.nan)
            buf[:, :self._buf.shape[1]] = self._buf
            self._buf = buf
            self.version += 1

    def has(self, key: Key) -> bool:
        return key in self.columns

    def keys(self) -> list[Key]:
        return [k for k in self.columns if k not in ("t", "step")]

    @property
    def capacity(self) -> int:
        return max(16, self.max_bytes // (8 * self.width))

    @property
    def bytes_used(self) -> int:
        return self.n * self.width * 8

    @property
    def effective_interval(self) -> int:
        return self.interval_steps * self.stride

    def wants(self, step: int) -> bool:
        """Whether the sample at this step number should be recorded."""
        return not self.full_stop and step % self.effective_interval == 0

    # ---- writing (engine thread) -------------------------------------------------------------------------
    def append(self, t: float, step: int, values: dict[Key, np.ndarray | float]) -> bool:
        with self._lock:
            if self.full_stop:
                return False
            if self.n >= self.capacity and not self._make_room(t, step):
                return False
            if self.n >= self._buf.shape[0]:
                grow = min(self.capacity, max(64, self._buf.shape[0] * 2))
                buf = np.full((grow, self.width), np.nan)
                buf[:self.n] = self._buf[:self.n]
                self._buf = buf
            row = self._buf[self.n]
            row[:] = np.nan
            row[0], row[1] = t, step
            for k, v in values.items():
                sl = self.columns.get(k)
                if sl is not None:
                    row[sl] = v
            self.n += 1
            self.version += 1
            return True

    def _make_room(self, t: float, step: int) -> bool:
        if self.overflow == "decimate":
            keep = self._buf[:self.n:2].copy()
            self._buf[:len(keep)] = keep
            self.n = len(keep)
            self.stride *= 2
            self.decimations += 1
            self._event(t, step, "recording", f"Memory limit reached: recording decimated, every "
                                              f"{self.effective_interval} steps from now on")
            return step % self.effective_interval == 0
        if self.overflow == "window":
            drop = max(1, self.n // 4)
            self._buf[:self.n - drop] = self._buf[drop:self.n].copy()
            self.n -= drop
            self.dropped += drop
            return True
        self.full_stop = True
        self._event(t, step, "recording", "Memory limit reached: recording stopped")
        return False

    # ---- events --------------------------------------------------------------------------------------------
    def _event(self, t: float, step: int, kind: str, message: str) -> None:
        self.events.append(Event(float(t), int(step), kind, message))
        if len(self.events) > 10_000:
            del self.events[:1000]
        self.version += 1

    def log(self, t: float, step: int, kind: str, message: str) -> None:
        with self._lock:
            self._event(t, step, kind, message)

    # ---- reading (any thread) -------------------------------------------------------------------------------
    def times(self) -> np.ndarray:
        return self.column("t")[:, 0]

    def steps(self) -> np.ndarray:
        return self.column("step")[:, 0].astype(np.int64)

    @property
    def generation(self) -> tuple[int, int]:
        """Changes whenever stored rows are removed or moved (decimation, window); appends keep it."""
        return self.decimations, self.dropped

    def column(self, key: Key, start: int = 0, stride: int = 1, stop: int | None = None) -> np.ndarray:
        """Copy of a column block, shape (n, width). Missing keys raise KeyError."""
        with self._lock:
            sl = self.columns[key]
            end = self.n if stop is None else min(stop, self.n)
            return self._buf[start:end:max(1, stride), sl].copy()

    def columns_for(self, keys: list[Key]) -> dict[Key, np.ndarray]:
        with self._lock:
            return {k: self._buf[:self.n, self.columns[k]].copy() for k in keys if k in self.columns}

    def event_list(self) -> list[Event]:
        with self._lock:
            return list(self.events)

    def object_ids(self) -> list[str]:
        return list(self.meta.get("objects", {}))

    def status_text(self) -> str:
        if self.n == 0:
            return "No samples recorded"
        t = self.times()
        mb = self.bytes_used / 2 ** 20
        extra = []
        if self.decimations:
            extra.append(f"decimated ×{self.stride}")
        if self.dropped:
            extra.append(f"{self.dropped:,} oldest samples dropped")
        if self.full_stop:
            extra.append("stopped at the memory limit")
        return (f"{self.n:,} samples · t = {t[0]:.6g}…{t[-1]:.6g} s · every {self.effective_interval} step(s) · "
                f"{mb:.1f} of {self.max_bytes / 2 ** 20:.0f} MB" + (f" · {', '.join(extra)}" if extra else ""))

    def index_at_time(self, t: float) -> int:
        """Index of the sample nearest to simulated time t (-1 if empty)."""
        ts = self.times()
        if not len(ts):
            return -1
        i = int(np.searchsorted(ts, t))
        if i <= 0:
            return 0
        if i >= len(ts):
            return len(ts) - 1
        return i if abs(ts[i] - t) < abs(ts[i - 1] - t) else i - 1


def finite_or_none(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None
