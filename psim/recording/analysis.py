"""Statistics of recorded variables and comparison between objects."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .variables import Evaluator, Variable


@dataclass
class Stats:
    count: int
    initial: float
    final: float
    minimum: float
    maximum: float
    mean: float           # time-weighted (trapezoidal) mean over the recorded interval
    rms: float            # time-weighted root mean square
    std: float            # time-weighted standard deviation
    t_min: float          # time at which the minimum occurs
    t_max: float

    def as_row(self) -> list[float]:
        return [self.initial, self.final, self.minimum, self.maximum, self.mean, self.rms, self.std]


STAT_COLUMNS = ["Initial", "Final", "Min", "Max", "Mean", "RMS", "Std. dev."]
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy < 2.0 only has trapz


def clean_row(values: list[float], rel: float = 1e-12) -> list[float]:
    """Replace floating-point noise (|x| below `rel` × the row's magnitude, at least `rel`) with 0 for display."""
    finite = [abs(v) for v in values if math.isfinite(v)]
    thr = rel * max(1.0, max(finite, default=0.0))
    return [0.0 if math.isfinite(v) and abs(v) < thr else v for v in values]


def _time_mean(t: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 1 or t[-1] <= t[0]:
        return float(np.mean(y))
    return float(_trapezoid(y, t) / (t[-1] - t[0]))


def statistics(t: np.ndarray, y: np.ndarray) -> Stats | None:
    """Time-weighted statistics of a sampled signal (samples may be unevenly spaced after decimation).

    NaN samples (values not available at that time) are ignored. Returns None if nothing is finite.
    """
    ok = np.isfinite(y) & np.isfinite(t)
    if not ok.any():
        return None
    t, y = t[ok], y[ok]
    i_min, i_max = int(np.argmin(y)), int(np.argmax(y))
    mean = _time_mean(t, y)
    rms = float(np.sqrt(max(_time_mean(t, y * y), 0.0)))
    var = max(_time_mean(t, (y - mean) ** 2), 0.0)
    return Stats(len(y), float(y[0]), float(y[-1]), float(y[i_min]), float(y[i_max]), mean, rms, float(np.sqrt(var)),
                 float(t[i_min]), float(t[i_max]))


def variable_statistics(ev: Evaluator, var: Variable) -> Stats | None:
    return statistics(ev.values(Variable.time()), ev.values(var))


def compare(ev: Evaluator, quantity: str, component: str, object_ids: list[str],
            frame: str | None = None) -> dict[str, Stats | None]:
    """The same variable for several objects (for side-by-side comparison)."""
    out = {}
    for oid in object_ids:
        try:
            out[oid] = variable_statistics(ev, Variable(oid, quantity, component, frame))
        except ValueError:
            out[oid] = None
    return out
