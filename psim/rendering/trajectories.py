"""Trajectory markers: world-space paths built from the recorded samples (no OpenGL here).

A path is recorded in a frame (global by default). A relative path — e.g. the
ball's path seen from a turntable — is drawn attached to the frame's *current*
pose, which is what an observer riding the frame sees. Paths are built
incrementally: each frame only the samples recorded since the last build are
processed, unless the recording was thinned (decimation / window) or the
settings changed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..coordinates.kinematics import rotation_of
from ..recording.recording import Recording
from ..recording.variables import Evaluator, Variable, VariableError
from ..scene.scene_manager import Scene, SceneError


@dataclass
class TrajectoryPath:
    object_id: str
    points: np.ndarray          # (m, 3) world coordinates
    times: np.ndarray           # (m,) simulated times
    color: list[float]
    width: float
    opacity: float
    arrows: bool = False
    arrow_spacing: float = 1.0
    time_labels: bool = False
    label_spacing: float = 1.0

    def segments(self) -> tuple[np.ndarray, np.ndarray]:
        """GL_LINES vertex pairs and colours for the polyline."""
        if len(self.points) < 2:
            return np.zeros((0, 3)), np.zeros((0, 4))
        pts = np.repeat(self.points, 2, axis=0)[1:-1]
        cols = np.tile([*self.color, self.opacity], (len(pts), 1))
        return pts, cols


@dataclass
class _Cache:
    key: tuple
    n_done: int = 0
    rel: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))


def thin(times: np.ndarray, interval: float, max_points: int, recent: bool) -> np.ndarray:
    """Indices of the samples to draw: at most one per `interval` (plus the last), at most `max_points`."""
    m = len(times)
    if m == 0:
        return np.zeros(0, int)
    idx = np.arange(m)
    if interval > 0:
        bins = np.floor((times - times[0]) / interval + 1e-9)
        keep = np.concatenate([[True], bins[1:] != bins[:-1]])
        keep[-1] = True
        idx = idx[keep]
    if len(idx) > max_points:
        if recent:
            idx = idx[-max_points:]
        else:  # thin evenly, keeping both ends
            idx = idx[np.unique(np.round(np.linspace(0, len(idx) - 1, max_points)).astype(int))]
    return idx


class TrajectoryBuilder:
    def __init__(self) -> None:
        self._cache: dict[str, _Cache] = {}
        self.clear_times: dict[str, float] = {}   # object id -> paths start after this simulated time

    def reset(self) -> None:
        self._cache.clear()
        self.clear_times.clear()

    def clear(self, object_id: str, rec: Recording | None) -> None:
        """'Clear trajectory': hide the path recorded so far (later samples still appear)."""
        if rec is not None and rec.n:
            self.clear_times[object_id] = float(rec.times()[-1])

    def build(self, scene: Scene, rec: Recording | None) -> list[TrajectoryPath]:
        if rec is None or rec.n == 0:
            return []
        out = []
        for obj in scene.objects.values():
            if not obj.trajectory_setting("enabled") or not scene.is_effectively_visible(obj.id):
                continue
            if not scene.simulating and not obj.trajectory_setting("preserve"):
                continue
            if not rec.has((obj.id, "origin")):
                continue
            path = self._path(scene, rec, obj)
            if path is not None:
                out.append(path)
        return out

    def _path(self, scene: Scene, rec: Recording, obj) -> TrajectoryPath | None:
        ts = obj.trajectory_setting
        frame = ts("frame") or None
        if frame and frame not in scene.objects and scene.frame(frame) is None:
            return None
        quantity = "position" if ts("point") == "origin" else "com"
        key = (id(rec), quantity, frame, rec.decimations, rec.dropped)
        c = self._cache.get(obj.id)
        if c is None or c.key != key or c.n_done > rec.n:
            c = self._cache[obj.id] = _Cache(key)
        if rec.n > c.n_done:
            try:
                ev = Evaluator(rec, scene, start=c.n_done)
                new = ev.vector(Variable(obj.id, quantity, frame=frame))
                new_t = ev.values(Variable.time())
            except (VariableError, SceneError):
                return None
            c.rel = np.vstack([c.rel, new])
            c.times = np.concatenate([c.times, new_t])
            c.n_done = ev.stop
        rel, times = c.rel, c.times
        ok = np.isfinite(rel).all(axis=1)
        t0 = self.clear_times.get(obj.id)
        if t0 is not None:
            ok &= times > t0
        recent = ts("history") == "recent"
        if recent and len(times):
            ok &= times >= times[-1] - ts("recent_duration")
        rel, times = rel[ok], times[ok]
        idx = thin(times, ts("interval"), ts("max_points"), recent)
        rel, times = rel[idx], times[idx]
        if frame:
            try:
                fp, fq = scene.reference_pose(frame)
            except SceneError:
                return None
            pts = np.asarray(fp, float) + rel @ rotation_of(fq).T
        else:
            pts = rel
        return TrajectoryPath(obj.id, pts, times, list(ts("color")), float(ts("width")), float(ts("opacity")),
                              bool(ts("arrows")), float(ts("arrow_spacing")), bool(ts("time_labels")),
                              float(ts("label_spacing")))


def marks(path: TrajectoryPath, spacing: float) -> list[int]:
    """Indices of the path points closest after each multiple of `spacing` (for arrows and time labels)."""
    if spacing <= 0 or len(path.times) < 2:
        return []
    t = path.times
    first = np.ceil(t[0] / spacing) * spacing
    targets = np.arange(first, t[-1] + 1e-12, spacing)
    if len(targets) > 500:   # keep the screen readable
        targets = targets[:: int(np.ceil(len(targets) / 500))]
    idx = np.unique(np.clip(np.searchsorted(t, targets - 1e-12), 0, len(t) - 1))
    return [int(i) for i in idx]
