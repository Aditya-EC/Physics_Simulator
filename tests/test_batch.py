"""Headless batch execution (psim.batch): exports, overrides, stop conditions, exit codes."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import pytest

from psim import batch
from psim.processes.process_base import TargetFilter
from psim.project import serialization as ser
from psim.units.unit_registry import UnitRegistry

G = 9.80665
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _project(scene, tmp_path, **sim):
    s, e = scene
    ball = e.add_new("body.point_mass", position=[0, 0, 20.0], name="Ball")
    e.set_global_parameters([{"name": "gz", "type": "float", "quantity": "acceleration", "value": -G}])
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    data = ser.ProjectData()
    for k, v in {"delta_time": 0.01, "duration": 1.0, "integrator": "rk4", **sim}.items():
        setattr(data.simulation, k, v)
    path = str(tmp_path / "p.psim")
    ser.save_project(path, s, UnitRegistry(), data)
    return path, ball


def _read(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], [[float(x) for x in r] for r in rows[1:]]


def test_batch_run_exports_csv_json_report_and_state(scene, tmp_path, capsys):
    path, ball = _project(scene, tmp_path)
    out = {k: str(tmp_path / f"out.{k}") for k in ("csv", "json", "html", "psimstate", "summary")}
    code = batch.main([path, "--csv", out["csv"], "--var", "Ball:position.z", "--var", f"{ball.id}:speed",
                       "--json", out["json"], "--report", out["html"], "--save-state", out["psimstate"],
                       "--summary-json", out["summary"]])
    assert code == batch.EXIT_OK
    head, rows = _read(out["csv"])
    assert head == ["time [s]", "Ball: Position z [m]", "Ball: Speed [m/s]"]
    t, z, v = rows[-1]
    assert t == pytest.approx(1.0) and z == pytest.approx(20 - G / 2, abs=1e-9) and v == pytest.approx(G, abs=1e-9)
    assert len(rows) == 101
    assert json.load(open(out["json"]))["recording"]["samples"] == 101
    assert "<html" in open(out["html"], encoding="utf-8").read()[:200].lower()
    assert json.load(open(out["psimstate"]))["clock"]["steps"] == 100
    summary = json.load(open(out["summary"]))
    assert summary["steps"] == 100 and summary["finished"] and summary["profile_ms_per_step"]["steps"] == 100
    assert "Finished at t = 1 s after 100 steps" in capsys.readouterr().out


def test_overrides_change_the_run_and_state_continues(scene, tmp_path):
    path, ball = _project(scene, tmp_path)
    csv1 = str(tmp_path / "a.csv")
    assert batch.main([path, "--quiet", "--duration", "0.5", "--set", "Ball.linear_velocity=[2,0,0]",
                       "--csv", csv1, "--var", "Ball:position"]) == 0
    _, rows = _read(csv1)
    assert rows[-1][1] == pytest.approx(1.0) and rows[-1][0] == pytest.approx(0.5)
    st = str(tmp_path / "half.psimstate")
    assert batch.main([path, "--quiet", "--duration", "0.5", "--save-state", st]) == 0
    csv2 = str(tmp_path / "b.csv")
    assert batch.main([path, "--quiet", "--restore-state", st, "--csv", csv2, "--var", "Ball:position.z"]) == 0
    _, rows = _read(csv2)
    assert rows[0][0] == pytest.approx(0.5) and rows[-1][1] == pytest.approx(20 - G / 2, abs=1e-9)


def test_global_override_and_stop_condition(scene, tmp_path):
    s, e = scene
    path, ball = _project(scene, tmp_path, duration=2.0, stop_conditions=[
        {"name": "Low", "expression": 'object("OBJ-000001").position.z < 15[m]', "action": "stop"}])
    summary = str(tmp_path / "s.json")
    assert batch.main([path, "--quiet", "--summary-json", summary]) == 0
    fired = json.load(open(summary))["stop_conditions"]
    assert fired[0]["name"] == "Low" and fired[0]["time"] == pytest.approx((2 * 5 / G) ** 0.5, abs=0.011)
    assert json.load(open(summary))["time"] == pytest.approx(fired[0]["time"])     # the run ended there
    assert batch.main([path, "--quiet", "--global", "gz=-1"]) == 0          # valid override
    assert batch.main([path, "--quiet", "--global", "gz=[1,2]"]) == batch.EXIT_INPUT


@pytest.mark.parametrize("args, message", [
    (["--var", "Ball:nonsense", "--csv", "x.csv"], "Unknown quantity"),
    (["--var", "Nobody:position", "--csv", "x.csv"], "No object 'Nobody'"),
    (["--set", "Ball.mass_input=-3"], "--set Ball.mass_input"),
    (["--global", "missing=1"], "No global parameter"),
    (["--dt", "-1"], "Invalid simulation settings"),
    (["--integrator", "magic"], "Unknown integrator"),
])
def test_invalid_input_exits_with_2(scene, tmp_path, capsys, args, message):
    path, _ = _project(scene, tmp_path)
    os.chdir(tmp_path)
    assert batch.main([path, "--quiet", *args]) == batch.EXIT_INPUT
    assert message in capsys.readouterr().err


def test_unreadable_project_and_numerical_failure(scene, tmp_path, capsys):
    bad = tmp_path / "bad.psim"
    bad.write_text("{not json")
    assert batch.main([str(bad)]) == batch.EXIT_INPUT
    s, e = scene
    a = e.add_new("body.point_mass", position=[0, 0, 0])
    b = e.add_new("body.point_mass", position=[2, 0, 0])
    e.assign_process("dyn.spring", TargetFilter(ids=[a.id, b.id]), {"stiffness": 1e30, "rest_length": 1.0})
    data = ser.ProjectData()
    data.simulation.delta_time, data.simulation.duration = 0.1, 100.0
    path = str(tmp_path / "unstable.psim")
    ser.save_project(path, s, UnitRegistry(), data)
    out = str(tmp_path / "partial.csv")
    assert batch.main([path, "--csv", out]) == batch.EXIT_FAILED
    assert "ERROR" in capsys.readouterr().err
    assert os.path.exists(out)                                    # partial results are still written


def test_module_entry_point_runs_as_a_process(tmp_path):
    r = subprocess.run([sys.executable, "-m", "psim.batch", "--help"], cwd=ROOT, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0 and "--restore-state" in r.stdout


def test_sweep_specs_grid_and_file_names():
    assert batch.parse_sweep("k=0:1:5")[1] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert batch.parse_sweep("k=2:2:1")[1] == [2.0]
    assert batch.parse_sweep("k=0.2:0.9:4")[1][-1] == 0.9                       # exact end
    assert batch.parse_sweep("Ball.restitution = 0.5, 0.9")[1] == [0.5, 0.9]
    assert batch.parse_sweep("Ball.linear_velocity=[[1,0,0],[0,2,0]]")[1] == [[1, 0, 0], [0, 2, 0]]
    for bad, msg in (("k", "NAME=VALUES"), ("k=1:2:0", "COUNT"), ("k=a:b:3", "numbers"), ("k=[1,", "JSON")):
        with pytest.raises(batch.BatchError, match=msg):
            batch.parse_sweep(bad)
    grid = batch.sweep_grid([("a", [1, 2]), ("b", ["x", "y", "z"])])
    assert grid[:4] == [{"a": 1, "b": "x"}, {"a": 1, "b": "y"}, {"a": 1, "b": "z"}, {"a": 2, "b": "x"}]
    with pytest.raises(batch.BatchError, match="only once"):
        batch.sweep_grid([("a", [1]), ("a", [2])])
    with pytest.raises(batch.BatchError, match="runs"):
        batch.sweep_grid([("a", list(range(1000))), ("b", list(range(1000)))])
    assert batch.run_path("out/run.csv", 7, 3) == "out/run_007.csv"
    assert batch.run_path("r{run}/x.json", 12, 2) == "r12/x.json"
    assert batch.run_path(None, 1, 1) is None


def test_sweep_runs_every_combination(scene, tmp_path, capsys):
    path, ball = _project(scene, tmp_path)
    table = str(tmp_path / "sweep.csv")
    per_run = str(tmp_path / "z.csv")
    code = batch.main([path, "--sweep", "Ball.position=[[0,0,20],[0,0,30]]", "--sweep", "Ball.linear_velocity="
                       "[[0,0,0],[0,0,1],[0,0,2]]", "--var", "Ball:position.z", "--sweep-csv", table,
                       "--csv", per_run])
    assert code == batch.EXIT_OK
    out = capsys.readouterr().out
    assert "Sweep of" in out and "6 runs" in out and "run 6/6" in out
    with open(table, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    head, rows = rows[0], rows[1:]
    assert head[:3] == ["run", "Ball.position", "Ball.linear_velocity"] and head[-1] == "Ball: Position z [m]"
    assert len(rows) == 6
    for r in rows:
        z0, vz = json.loads(r[1])[2], json.loads(r[2])[2]
        assert r[head.index("finished")] == "True" and r[head.index("error")] == ""
        assert float(r[-1]) == pytest.approx(z0 + vz - G / 2, abs=1e-9)             # z(1 s) of each run
    for k in range(1, 7):
        _, series = _read(str(tmp_path / f"z_{k}.csv"))
        assert len(series) == 101                                                    # one file per run


def test_sweep_in_parallel_matches_serial(scene, tmp_path):
    path, _ = _project(scene, tmp_path)
    args = ["--quiet", "--sweep", "Ball.linear_velocity=[[0,0,0],[1,0,3],[2,0,-1]]", "--var", "Ball:position"]
    a, b = str(tmp_path / "a.csv"), str(tmp_path / "b.csv")
    assert batch.main([path, *args, "--sweep-csv", a]) == 0
    assert batch.main([path, *args, "--sweep-csv", b, "--jobs", "2"]) == 0
    with open(a, encoding="utf-8") as fa, open(b, encoding="utf-8") as fb:
        ra, rb = list(csv.reader(fa)), list(csv.reader(fb))
    wall = ra[0].index("wall [s]")
    assert [r[:wall] + r[wall + 1:] for r in ra] == [r[:wall] + r[wall + 1:] for r in rb]    # identical results


def test_sweep_reports_bad_runs_and_bad_input(scene, tmp_path, capsys):
    path, _ = _project(scene, tmp_path)
    table = str(tmp_path / "t.csv")
    # one invalid value: that run is reported, the others still run
    assert batch.main([path, "--quiet", "--sweep", "Ball.mass_input=1,-3,2", "--sweep-csv", table]) == \
        batch.EXIT_INPUT
    with open(table, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    err = rows[0].index("error")
    assert [bool(r[err]) for r in rows[1:]] == [False, True, False] and "mass_input" in rows[2][err]
    assert "run 2:" in capsys.readouterr().err
    assert batch.main([path, "--quiet", "--sweep", "gz=-9.8,-1.6"]) == 0          # global parameters
    assert batch.main([path, "--quiet", "--sweep", "nothing=1,2"]) == batch.EXIT_INPUT
    assert batch.main([path, "--quiet", "--sweep", "k=1:2:0"]) == batch.EXIT_INPUT
    assert batch.main([path, "--quiet", "--jobs", "2"]) == batch.EXIT_INPUT         # --jobs needs --sweep
