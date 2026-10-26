"""Variables computed from recordings, statistics, CSV/JSON export and the HTML report."""
from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from psim import numerics
from psim.processes.process_base import TargetFilter
from psim.recording.analysis import compare, statistics, variable_statistics
from psim.recording.export import export_csv, export_json, export_trajectory_csv, generate_report
from psim.recording.recording import Recording
from psim.recording.variables import Evaluator, Variable, VariableError, label, object_quantities

from test_recording import make_engine, run


def _turntable(scene):
    s, e = scene
    deck = e.add_new("basic.cylinder")
    deck.name = "Deck"
    e.assign_process("rot.fixed_axis", TargetFilter(ids=[deck.id]), {"axis": [0, 0, 1], "angular_speed": 1.5})
    ball = e.add_new("body.point_mass", position=[-2, 0.5, 0.3])
    ball.name = "Ball"
    ball.linear_velocity = [1.0, 0.2, 0.0]
    e.assign_process("kin.free_motion", TargetFilter(ids=[ball.id]))
    fid = e.add_frame("Rim", deck.id, [0.7, 0.0, 0.1], numerics.quat_from_axis_angle([1, 0, 0], 0.3))
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.001, duration=2.0, integrator="rk4"))
    return s, deck, ball, fid, rec


def test_relative_variables_match_finite_differences(scene):
    s, deck, ball, fid, rec = _turntable(scene)
    ev = Evaluator(rec, s)
    t = ev.values(Variable.time())
    for frame in (fid, deck.id):
        pos = ev.vector(Variable(ball.id, "com", frame=frame))
        vel = ev.vector(Variable(ball.id, "velocity", frame=frame))
        acc = ev.vector(Variable(ball.id, "acceleration", frame=frame))
        h = t[2:] - t[:-2]
        np.testing.assert_allclose(vel[1:-1], (pos[2:] - pos[:-2]) / h[:, None], atol=1e-5)
        np.testing.assert_allclose(acc[1:-1], (vel[2:] - vel[:-2]) / h[:, None], atol=1e-4)
    # global quantities: straight line at constant velocity
    np.testing.assert_allclose(ev.values(Variable(ball.id, "acceleration", "mag")), 0, atol=1e-12)
    np.testing.assert_allclose(ev.values(Variable(ball.id, "distance")), math.hypot(1.0, 0.2) * t, atol=1e-9)
    np.testing.assert_allclose(ev.vector(Variable(ball.id, "displacement")), np.outer(t, [1.0, 0.2, 0]), atol=1e-9)
    # relative rotation: the deck seen from its own rim frame does not rotate
    np.testing.assert_allclose(ev.vector(Variable(deck.id, "angular_velocity", frame=fid)), 0, atol=1e-12)
    eul = ev.vector(Variable(deck.id, "euler"))
    q = rec.column((deck.id, "q"))
    np.testing.assert_allclose(eul[::200], [numerics.quat_to_euler(x) for x in q[::200]], atol=1e-12)


def test_dynamic_quantities_and_process_outputs(scene):
    s, e = scene
    cube = e.add_new("basic.cube")          # 1 m cube of the default density
    cube.torque = [0, 0, 2.0]
    cube.force = [3.0, 0, 0]
    g = e.assign_process("dyn.gravity", TargetFilter(ids=[cube.id]))
    f = e.assign_process("dyn.applied_force", TargetFilter(ids=[cube.id]))
    rec = Recording(interval_steps=10)
    run(make_engine(scene, rec, delta_time=0.01, duration=1.0, integrator="rk4"))
    ev = Evaluator(rec, s)
    m = cube.mass()
    np.testing.assert_allclose(ev.vector(Variable(cube.id, "net_force")), np.tile([3.0, 0, -9.80665 * m], (rec.n, 1)),
                               atol=1e-9)
    np.testing.assert_allclose(ev.vector(Variable(cube.id, f"proc:{g.uid}:force")),
                               np.tile([0, 0, -9.80665 * m], (rec.n, 1)), atol=1e-9)
    np.testing.assert_allclose(ev.vector(Variable(cube.id, f"proc:{f.uid}:torque")), np.tile([0, 0, 2.0], (rec.n, 1)),
                               atol=1e-9)
    np.testing.assert_allclose(ev.vector(Variable(cube.id, "net_torque")), np.tile([0, 0, 2.0], (rec.n, 1)), atol=1e-9)
    t = ev.values(Variable.time())
    i_zz = 2.0 / ev.vector(Variable(cube.id, "angular_acceleration"))[0, 2]
    ke = ev.values(Variable(cube.id, "kinetic_energy"))
    v = ev.vector(Variable(cube.id, "velocity"))
    np.testing.assert_allclose(ke, 0.5 * m * np.sum(v * v, 1) + 0.5 * i_zz * (2.0 / i_zz * t) ** 2, rtol=1e-9)
    assert f"proc:{g.uid}:force" in object_quantities(rec, cube.id)
    assert label(Variable(cube.id, "velocity", "x", "FRM-0001"), {cube.id: "Cube"}, {"FRM-0001": "Lab"}) \
        == "Cube: Velocity x rel. Lab"
    with pytest.raises(VariableError, match="not recorded"):
        ev.values(Variable("OBJ-999999", "velocity", "x"))


def test_statistics_are_time_weighted():
    t = np.linspace(0, 2 * np.pi, 2001)
    st = statistics(t, np.sin(t))
    assert st.mean == pytest.approx(0, abs=1e-9) and st.rms == pytest.approx(1 / math.sqrt(2), rel=1e-6)
    assert st.maximum == pytest.approx(1) and st.t_max == pytest.approx(np.pi / 2, abs=1e-2)
    # uneven sampling (e.g. after decimation) must not bias the mean
    t2 = np.concatenate([np.linspace(0, 1, 1000, endpoint=False), np.linspace(1, 2, 11)])
    y2 = np.where(t2 < 1, 0.0, 1.0)
    assert statistics(t2, y2).mean == pytest.approx(0.5, abs=2e-3)
    assert statistics(t, np.full_like(t, np.nan)) is None


def test_compare_objects(scene):
    s, deck, ball, fid, rec = _turntable(scene)
    ev = Evaluator(rec, s)
    table = compare(ev, "speed", "", [deck.id, ball.id])
    assert table[ball.id].mean == pytest.approx(math.hypot(1.0, 0.2))
    assert table[deck.id].maximum == pytest.approx(0, abs=1e-12)
    assert variable_statistics(ev, Variable(ball.id, "position", "x")).final == pytest.approx(0.0, abs=1e-9)


def test_exports_and_report(scene, tmp_path):
    from psim.units.unit_registry import UnitRegistry
    s, deck, ball, fid, rec = _turntable(scene)
    rec.log(1.0, 1000, "warning", "something <odd> happened")
    ev = Evaluator(rec, s)
    units = UnitRegistry()
    units.set_preferred_unit("velocity", "km/h")
    path = tmp_path / "ball.csv"
    n = export_csv(str(path), ev, [Variable(ball.id, "velocity", "x"), Variable(ball.id, "position", "y", fid)], units,
                   {ball.id: "Ball"}, {fid: "Rim"})
    rows = list(csv.reader(open(path, encoding="utf-8")))
    assert n == rec.n and len(rows) == rec.n + 1
    assert rows[0] == ["time [s]", "Ball: Velocity x [km/h]", "Ball: Position y rel. Rim [m]"]
    assert float(rows[1][1]) == pytest.approx(3.6)
    export_trajectory_csv(str(tmp_path / "traj.csv"), ev, ball.id, fid)
    assert len(list(csv.reader(open(tmp_path / "traj.csv")))) == rec.n + 1
    export_json(str(tmp_path / "all.json"), rec)
    data = json.load(open(tmp_path / "all.json"))
    assert data["recording"]["samples"] == rec.n and len(data["time"]) == rec.n
    assert f"{ball.id}/origin" in data["channels"] and data["events"][-1]["kind"] == "warning"
    report = tmp_path / "report.html"
    generate_report(str(report), rec, s, units, "Turntable run", images=[("Speed plot", b"\x89PNG\r\n\x1a\nfake")])
    text = report.read_text(encoding="utf-8")
    assert "Turntable run" in text and "Ball" in text and "Kinetic energy" in text and "km/h" in text
    assert "something &lt;odd&gt; happened" in text and "data:image/png;base64," in text


def test_evaluator_reads_a_consistent_snapshot():
    """Columns read at different moments of a growing recording must have the same length."""
    rec = Recording(max_bytes=10_000, overflow="decimate")
    rec.ensure_columns([(("o", "v"), 3), (("o", "w"), 3)])
    for k in range(10):
        rec.append(k * 0.1, k, {("o", "v"): [k, 0, 0], ("o", "w"): [0, 0, k]})
    ev = Evaluator(rec)
    v = ev._col(("o", "v"))
    rec.append(1.0, 10, {("o", "v"): [10, 0, 0], ("o", "w"): [0, 0, 10]})   # the simulation keeps recording
    w = ev._col(("o", "w"))
    assert len(v) == len(w) == 10 and ev.n == 10
    for k in range(11, 2000):          # force a decimation: earlier rows move
        rec.append(k * 0.1, k, {("o", "v"): [k, 0, 0], ("o", "w"): [0, 0, k]})
    with pytest.raises(VariableError, match="thinned"):
        ev._col("step")                # not read before: the snapshot can no longer be completed
