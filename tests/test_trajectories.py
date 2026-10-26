"""Trajectory paths from recordings, vector arrows and object label settings."""
from __future__ import annotations

import numpy as np
import pytest

from psim.processes.process_base import TargetFilter
from psim.recording.recording import Recording
from psim.rendering.trajectories import TrajectoryBuilder, TrajectoryPath, marks, thin

from test_recording import make_engine


def test_thinning():
    t = np.arange(0, 10.001, 0.01)
    idx = thin(t, 0.5, 10_000, False)
    assert idx[0] == 0 and idx[-1] == len(t) - 1 and len(idx) == 21   # one per 0.5 s bin, 0 … 10 s
    idx = thin(t, 0.0, 100, False)
    assert len(idx) == 100 and idx[0] == 0 and idx[-1] == len(t) - 1
    idx = thin(t, 0.0, 100, True)
    assert list(idx) == list(range(len(t) - 100, len(t)))
    assert len(thin(np.zeros(0), 1.0, 10, False)) == 0


def _spinning_scene(scene):
    s, e = scene
    deck = e.add_new("basic.cylinder")
    e.assign_process("rot.fixed_axis", TargetFilter(ids=[deck.id]), {"axis": [0, 0, 1], "angular_speed": 1.0})
    ball = e.add_new("body.point_mass", position=[-2, 0.3, 0.2])
    ball.linear_velocity = [1.0, 0.0, 0.0]
    e.assign_process("kin.free_motion", TargetFilter(ids=[ball.id]))
    e.set_property([ball.id], "trajectory.enabled", True)
    return s, e, s.get(deck.id), s.get(ball.id)


def test_global_path_is_built_incrementally(scene):
    s, e, deck, ball = _spinning_scene(scene)
    rec = Recording()
    eng = make_engine(scene, rec, delta_time=0.01, duration=2.0)
    builder = TrajectoryBuilder()
    for _ in range(50):
        eng.step()
    first = builder.build(s, rec)
    assert len(first) == 1 and len(first[0].points) == 50
    while eng.step():
        pass
    incremental = builder.build(s, rec)[0]
    fresh = TrajectoryBuilder().build(s, rec)[0]
    np.testing.assert_array_equal(incremental.points, fresh.points)
    np.testing.assert_allclose(incremental.points, rec.column((ball.id, "com")))
    # settings: interval and recent history
    e.set_property([ball.id], "trajectory.interval", 0.25)
    path = builder.build(s, rec)[0]
    assert len(path.points) == 9 and path.times[-1] == pytest.approx(2.0)
    e.set_property([ball.id], "trajectory.history", "recent")
    e.set_property([ball.id], "trajectory.recent_duration", 0.5)
    assert builder.build(s, rec)[0].times[0] >= 1.5 - 1e-9
    # clear: only later samples are drawn
    builder.clear(ball.id, rec)
    assert len(builder.build(s, rec)[0].points) == 0     # everything so far is hidden
    # hidden after stopping unless preserved
    builder.clear_times.clear()
    e.set_property([ball.id], "trajectory.preserve", False)
    assert builder.build(s, rec) == []
    s.simulating = True
    assert len(builder.build(s, rec)) == 1
    s.simulating = False


def test_relative_path_attached_to_the_frame(scene):
    from psim import numerics
    s, e, deck, ball = _spinning_scene(scene)
    e.set_property([ball.id], "trajectory.frame", deck.id)
    rec = Recording()
    eng = make_engine(scene, rec, delta_time=0.01, duration=1.5, integrator="rk4")
    while eng.step():
        pass
    # put the deck where the simulation left it: the newest path point must coincide with the ball
    q_end = rec.column((deck.id, "q"))[-1]
    s.get(deck.id).set_orientation(q_end)
    path = TrajectoryBuilder().build(s, rec)[0]
    np.testing.assert_allclose(path.points[-1], rec.column((ball.id, "com"))[-1], atol=1e-9)
    # and earlier points are rotated with the deck: the first one is the start point seen from the turned deck
    r = np.asarray(numerics.quat_to_matrix(q_end))
    np.testing.assert_allclose(path.points[0], r @ np.array([-2, 0.3, 0.2]), atol=1e-9)
    # a decimated recording invalidates the cache
    builder = TrajectoryBuilder()
    n1 = len(builder.build(s, rec)[0].points)
    rec.max_bytes = rec.width * 8 * 100
    rec.append(rec.times()[-1] + 1, rec.steps()[-1] + 1, {})
    assert len(builder.build(s, rec)[0].points) < n1


def test_marks_and_segments():
    t = np.linspace(0, 3, 301)
    path = TrajectoryPath("x", np.c_[t, np.zeros_like(t), np.zeros_like(t)], t, [1, 0, 0], 2.0, 0.5)
    assert [round(path.times[i], 6) for i in marks(path, 1.0)] == [0.0, 1.0, 2.0, 3.0]
    pts, cols = path.segments()
    assert len(pts) == 2 * 300 and cols[0][3] == 0.5


def test_vector_arrows_and_label_settings(scene):
    from psim.rendering.overlays import object_label_lines
    from psim.rendering.vector_arrows import arrow_lines, arrow_vectors
    from psim.units.unit_registry import UnitRegistry
    s, e = scene
    o = e.add_new("body.point_mass", position=[1, 2, 3])
    for path, value in (("linear_velocity", [2, 0, 0]), ("overlay.velocity_arrow", True), ("overlay.velocity", True),
                        ("arrow_scale.velocity", 0.5), ("label_unit.velocity", "km/h"),
                        ("overlay.angular_acceleration", True), ("angular_acceleration", [0, 0, 3])):
        e.set_property([o.id], path, value)
    o = s.get(o.id)
    arrows = arrow_vectors(s, o)
    assert len(arrows) == 1 and arrows[0][0] == "velocity"
    np.testing.assert_allclose(arrows[0][1], [1, 2, 3])
    np.testing.assert_allclose(arrows[0][2], [1.0, 0, 0])
    pts, cols = arrow_lines(arrows[0][1], arrows[0][2], np.array([0, 0, -1.0]), (0, 1, 0))
    np.testing.assert_allclose(pts[1], [2, 2, 3])
    assert len(pts) == 6
    lines = object_label_lines(s, o, UnitRegistry())
    assert "v = (7.200, 0.000, 0.000) km/h" in lines and any(x.startswith("α = ") for x in lines)
    s.runtime[o.id] = {"angular_acceleration": np.array([0, 0, 5.0]), "acceleration": None}
    assert any("5.000" in x for x in object_label_lines(s, o, UnitRegistry()) if x.startswith("α"))
    e.set_property([o.id], "label_unit.velocity", "bogus")   # an invalid unit falls back to the project unit
    assert "v = (2.000, 0.000, 0.000) m/s" in object_label_lines(s, s.get(o.id), UnitRegistry())
