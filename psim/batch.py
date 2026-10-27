"""Headless batch execution: run a project without the GUI and export the results.

    python -m psim.batch project.psim [options]        (also installed as the `psim-batch` command)

The run uses the project's simulation settings (overridable on the command line), runs as fast as
possible on the calling thread, and writes any of: a CSV of selected variables, the full recording
as JSON, the HTML report and the final simulation state. Global parameters and object properties
can be overridden for parameter studies:

    python -m psim.batch drop.psim --duration 5 --global drag_cd=0.8 \\
        --set OBJ-000002.linear_velocity=[3,0,4] --csv out.csv --var "Ball:position" --var "Ball:speed"

Parameter sweeps run the project once per combination of values (a grid when --sweep is repeated) and write
one row per run — the swept values, how the run ended, the stop conditions that fired and the final value of
every --var — to --sweep-csv (or print them). Per-run files get the run number before their extension (or in
place of "{run}"). --jobs runs several in parallel processes:

    python -m psim.batch drop.psim --sweep drag_cd=0.2:1.0:5 --sweep "Ball.restitution=0.5,0.9" \
        --var Ball:position.z --sweep-csv sweep.csv --jobs 4

Values are START:STOP:COUNT (evenly spaced, both ends included), a comma-separated list, or a JSON list
(for vectors: '[[1,0,0],[2,0,0]]'). A name with a dot is an object property (OBJECT.PATH), else a global
parameter.

Exit status: 0 success, 2 invalid input (project, settings, plan, variables), 3 the simulation failed
(numerical failure or process error; the partial results are still written), 130 interrupted. A sweep
reports the worst status of its runs.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

EXIT_OK, EXIT_INPUT, EXIT_FAILED, EXIT_INTERRUPTED = 0, 2, 3, 130


class BatchError(Exception):
    """Invalid batch input (reported with exit status 2)."""


@dataclass
class BatchResult:
    steps: int = 0
    time: float = 0.0
    wall_seconds: float = 0.0
    finished: bool = False
    error: str = ""
    fired: list = field(default_factory=list)          # (name, action, time) of stop conditions
    profile: dict = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)    # (level, message)
    outputs: list = field(default_factory=list)        # written files
    final: dict = field(default_factory=dict)          # "--var" column header → value at the last sample

    def to_dict(self) -> dict:
        return {"steps": self.steps, "time": self.time, "wall_seconds": self.wall_seconds,
                "finished": self.finished, "error": self.error,
                "stop_conditions": [{"name": n, "action": a, "time": t} for n, a, t in self.fired],
                "profile_ms_per_step": self.profile, "diagnostics": [list(d) for d in self.diagnostics],
                "outputs": self.outputs, "final": self.final}


# ---- input helpers -------------------------------------------------------------------------------------------------

def _value(text: str):
    """A command-line value: JSON (numbers, lists, true/false, strings in quotes) or else a plain string."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def _resolve_object(scene, ref: str) -> str:
    if ref in scene.objects:
        return ref
    matches = [o.id for o in scene.objects.values() if o.name == ref]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise BatchError(f"Object name '{ref}' is ambiguous ({', '.join(matches)}); use the object ID")
    raise BatchError(f"No object '{ref}' in the project")


def apply_overrides(scene, editor, globals_: list[str], props: list[str]) -> None:
    """--global name=value and --set OBJECT.path=value (validated like edits in the inspector)."""
    from .processes.global_parameters import validate_globals
    from .properties.property_system import PropertyError
    from .scene.scene_manager import SceneError
    params = [dict(g) for g in scene.global_parameters]
    for item in globals_:
        name, sep, text = item.partition("=")
        if not sep:
            raise BatchError(f"--global needs name=value, got '{item}'")
        g = next((g for g in params if g["name"] == name.strip()), None)
        if g is None:
            raise BatchError(f"No global parameter '{name.strip()}' in the project")
        g["value"] = _value(text)
    if globals_:
        try:
            editor.set_global_parameters(validate_globals(params))
        except Exception as exc:   # ProcessError with the parameter name
            raise BatchError(str(exc)) from None
    for item in props:
        target, sep, text = item.partition("=")
        ref, dot, path = target.partition(".")
        if not sep or not dot:
            raise BatchError(f"--set needs OBJECT.property=value, got '{item}'")
        oid = _resolve_object(scene, ref.strip())
        try:
            editor.set_property([oid], path.strip(), _value(text))
        except (PropertyError, SceneError) as exc:
            raise BatchError(f"--set {target}: {exc}") from None


def parse_variable(scene, spec: str):
    """OBJECT:QUANTITY[.COMPONENT][@FRAME] → Variables (vector quantities without a component give x, y, z)."""
    from .recording.variables import QUANTITIES, VECTOR, Variable
    obj, sep, rest = spec.partition(":")
    if not sep:
        raise BatchError(f"Variable '{spec}' must be OBJECT:QUANTITY")
    rest, _, frame = rest.partition("@")
    quantity, _, comp = rest.partition(".")
    quantity = quantity.strip()
    if quantity not in QUANTITIES and not quantity.startswith("custom."):
        raise BatchError(f"Unknown quantity '{quantity}' (known: {', '.join(q for q in QUANTITIES if q != 'step')})")
    oid = _resolve_object(scene, obj.strip())
    frame_ref = None
    if frame.strip():
        f = frame.strip()
        frame_ref = f if scene.frame(f) is not None else _resolve_object(scene, f)
    info = QUANTITIES.get(quantity)
    vector = bool(info and info.vector)
    if comp and (not vector or comp not in VECTOR):
        raise BatchError(f"Variable '{spec}': component must be one of {', '.join(VECTOR)} for a vector quantity")
    comps = [comp] if comp else (["x", "y", "z"] if vector else [""])
    return [Variable(oid, quantity, c, frame_ref) for c in comps]


# ---- the run --------------------------------------------------------------------------------------------------

def run_batch(project: str, *, duration: float | None = None, delta_time: float | None = None,
              integrator: str | None = None, record_every: int | None = None, globals_: list[str] = (),
              props: list[str] = (), variables: list[str] = (), csv_path: str | None = None,
              json_path: str | None = None, report_path: str | None = None, state_out: str | None = None,
              state_in: str | None = None, max_wall: float | None = None, log=print,
              final_values: bool = False) -> BatchResult:
    """One headless run. With `final_values`, the --var variables are recorded and their last values (display
    units) returned in BatchResult.final."""
    from .project.serialization import ProjectError, load_project, write_json_atomic
    from .recording.export import export_csv, export_json, generate_report
    from .recording.variables import Evaluator, Variable, VariableError, recorded_objects
    from .scene.editing import SceneEditor
    from .scene.scene_manager import Scene
    from .simulation.factory import build_engine, plan_run
    from .simulation.settings import SettingsError
    from .simulation.simulation_engine import SimulationError
    from .units.unit_registry import UnitRegistry

    res = BatchResult()
    scene, units = Scene(), UnitRegistry()
    try:
        data = load_project(project, scene, units)
    except (ProjectError, OSError) as exc:
        raise BatchError(f"Cannot open {project}: {exc}") from None
    for w in data.warnings:
        res.diagnostics.append(("warning", w))
    editor = SceneEditor(scene)
    apply_overrides(scene, editor, list(globals_), list(props))
    s = data.simulation
    if duration is not None:
        s.duration = float(duration)
    if delta_time is not None:
        s.delta_time = float(delta_time)
    if integrator is not None:
        s.integrator = integrator
    if record_every is not None:
        s.record_every_n_steps = int(record_every)
    want_record = bool(csv_path or json_path or report_path or (final_values and variables))
    if want_record:
        s.record_enabled = True
    try:
        s.check()
    except SettingsError as exc:
        raise BatchError(f"Invalid simulation settings: {exc}") from None
    plan = plan_run(scene, s)
    try:
        plan, eng, rec = build_engine(scene, s, record=want_record, plan=plan)
    except SimulationError as exc:
        raise BatchError(str(exc)) from None
    res.diagnostics += [(d.level, d.message) for d in plan.diagnostics]
    if eng is None:
        raise BatchError("; ".join(d.message for d in plan.errors))
    variables_parsed = [v for spec in variables for v in parse_variable(scene, spec)]
    if state_in:
        try:
            with open(state_in, encoding="utf-8") as fh:
                eng.import_state(json.load(fh))
        except (OSError, ValueError, SimulationError) as exc:
            raise BatchError(f"Cannot restore {state_in}: {exc}") from None
    log(f"Running {project}: {len(plan.body_ids)} bodies, Δt = {s.delta_time:g} s, duration {s.duration:g} s, "
        f"{s.integrator.replace('_', ' ')}")
    start = time.perf_counter()
    try:
        while eng.step():
            res.fired += eng.take_fired()
            if max_wall is not None and time.perf_counter() - start > max_wall:
                res.error = f"Stopped after the wall-clock limit of {max_wall:g} s"
                break
    except KeyboardInterrupt:
        res.error = "Interrupted"
        raise
    except Exception as exc:   # numerical failure or process error: report, keep the partial results
        res.error = str(exc)
    finally:
        res.wall_seconds = time.perf_counter() - start
        res.steps, res.time, res.finished = eng.clock.steps, eng.clock.time, eng.finished
        res.profile = {k: round(v, 6) if isinstance(v, float) else v for k, v in eng.profile_report().items()}
    if rec is not None:
        if eng.sampler is not None and not eng.finished:
            eng.sampler.final()
        if final_values and variables_parsed:
            from .recording.export import series_table
            names = {o.id: o.name for o in scene.objects.values()}
            try:
                headers, cols = series_table(Evaluator(rec, scene), variables_parsed, units, names)
            except (VariableError, KeyError) as exc:
                raise BatchError(f"Final values: {exc}") from None
            res.final = {h: (float(c[-1]) if len(c) else math.nan) for h, c in zip(headers[1:], cols[1:])}
        if csv_path:
            if not variables_parsed:
                variables_parsed = [Variable(oid, q, c) for oid in recorded_objects(rec)
                                    for q in ("position", "velocity") for c in "xyz"]
            names = {o.id: o.name for o in scene.objects.values()}
            try:
                n = export_csv(csv_path, Evaluator(rec, scene), variables_parsed, units, names)
            except (VariableError, KeyError) as exc:
                raise BatchError(f"CSV export: {exc}") from None
            res.outputs.append(csv_path)
            log(f"Wrote {n} samples to {csv_path}")
        if json_path:
            export_json(json_path, rec, {o.id: o.name for o in scene.objects.values()})
            res.outputs.append(json_path)
        if report_path:
            generate_report(report_path, rec, scene, units, title=data.metadata.get("title") or "Batch run")
            res.outputs.append(report_path)
    if state_out:
        write_json_atomic(state_out, eng.export_state())
        res.outputs.append(state_out)
    return res


# ---- parameter sweeps ------------------------------------------------------------------------------------------

MAX_SWEEP_RUNS = 100_000


def parse_sweep(spec: str) -> tuple[str, list]:
    """NAME=VALUES → (NAME, values). VALUES: START:STOP:COUNT, a comma list, or a JSON list."""
    name, sep, text = spec.partition("=")
    name, text = name.strip(), text.strip()
    if not sep or not name or not text:
        raise BatchError(f"--sweep needs NAME=VALUES, got '{spec}'")
    if text.startswith("["):
        try:
            values = json.loads(text)
        except ValueError:
            raise BatchError(f"--sweep {name}: '{text}' is not a JSON list") from None
        if not isinstance(values, list):
            raise BatchError(f"--sweep {name}: expected a list of values")
    elif text.count(":") == 2:
        try:
            a, b, n = (float(x) for x in text.split(":"))
        except ValueError:
            raise BatchError(f"--sweep {name}: START:STOP:COUNT needs numbers, got '{text}'") from None
        if not all(math.isfinite(x) for x in (a, b, n)) or n < 1 or n != int(n):
            raise BatchError(f"--sweep {name}: COUNT must be a positive integer and the ends finite")
        n = int(n)
        values = [a + (b - a) * k / (n - 1) for k in range(n)] if n > 1 else [a]
        values[-1] = b if n > 1 else a           # the end exactly (not a + (b − a) with rounding)
    else:
        values = [_value(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise BatchError(f"--sweep {name}: no values")
    return name, values


def sweep_grid(sweeps: list[tuple[str, list]]) -> list[dict]:
    """Every combination of the sweep values ({name: value}); the first sweep varies slowest."""
    names = [n for n, _ in sweeps]
    if len(set(names)) != len(names):
        raise BatchError("Each name can be swept only once")
    count = math.prod(len(v) for _, v in sweeps)
    if count > MAX_SWEEP_RUNS:
        raise BatchError(f"The sweep has {count} runs (at most {MAX_SWEEP_RUNS})")
    return [dict(zip(names, combo)) for combo in itertools.product(*[v for _, v in sweeps])]


def run_path(path: str | None, run: int, width: int) -> str | None:
    """A per-run file name: {run} is replaced by the run number, else it is inserted before the extension."""
    if not path:
        return path
    tag = f"{run:0{width}d}"
    if "{run}" in path:
        return path.replace("{run}", tag)
    root, ext = os.path.splitext(path)
    return f"{root}_{tag}{ext}"


def _sweep_run(args: tuple) -> tuple[int, dict]:
    """One run of a sweep (module level, so that worker processes can call it)."""
    index, project, point, kwargs = args
    globals_ = list(kwargs.pop("globals_", []))
    props = list(kwargs.pop("props", []))
    for name, value in point.items():
        (props if "." in name else globals_).append(f"{name}={json.dumps(value)}")
    row = {"status": EXIT_OK}
    try:
        res = run_batch(project, globals_=globals_, props=props, log=lambda *_a, **_k: None, final_values=True,
                        **kwargs)
    except BatchError as exc:
        return index, {"status": EXIT_INPUT, "error": str(exc)}
    row.update(res.to_dict())
    if res.error:
        row["status"] = EXIT_FAILED
    return index, row


def run_sweep(project: str, sweeps: list[str], jobs: int = 1, sweep_csv: str | None = None, log=print,
              **kwargs) -> tuple[list[dict], list[dict]]:
    """Run every combination of the sweeps; returns (points, rows) in run order and writes sweep_csv."""
    points = sweep_grid([parse_sweep(sp) for sp in sweeps])
    width = len(str(len(points)))
    tasks = []
    for k, point in enumerate(points, 1):
        kw = dict(kwargs)
        for key in ("csv_path", "json_path", "report_path", "state_out"):
            kw[key] = run_path(kw.get(key), k, width)
        tasks.append((k, project, point, kw))
    log(f"Sweep of {project}: {len(points)} runs" + (f" in {jobs} processes" if jobs > 1 else ""))
    rows: dict[int, dict] = {}

    def report(index, row):
        rows[index] = row
        values = ", ".join(f"{n} = {json.dumps(v)}" for n, v in points[index - 1].items())
        end = row.get("error") or (f"t = {row['time']:.6g} s" + ("" if row.get("finished") else " (stopped)"))
        log(f"  run {index}/{len(points)}: {values} → {end}")

    if jobs > 1 and len(tasks) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            for index, row in pool.map(_sweep_run, tasks):
                report(index, row)
    else:
        for task in tasks:
            report(*_sweep_run(task))
    ordered = [rows[k] for k in range(1, len(points) + 1)]
    if sweep_csv:
        write_sweep_csv(sweep_csv, points, ordered)
    return points, ordered


def write_sweep_csv(path: str, points: list[dict], rows: list[dict]) -> None:
    names = list(points[0]) if points else []
    finals = list(dict.fromkeys(h for r in rows for h in r.get("final", {})))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", *names, "finished", "steps", "time [s]", "wall [s]", "stop conditions", "error",
                    *finals])
        for k, (point, row) in enumerate(zip(points, rows), 1):
            fired = "; ".join(f"{c['name']} @ {c['time']:.6g} s" for c in row.get("stop_conditions", []))
            values = [v if isinstance(v, (int, float)) and not isinstance(v, bool) else json.dumps(v)
                      for v in point.values()]
            final = [row.get("final", {}).get(h, "") for h in finals]
            w.writerow([k, *values, row.get("finished", False), row.get("steps", 0),
                        repr(float(row.get("time", 0.0))), round(float(row.get("wall_seconds", 0.0)), 4), fired,
                        row.get("error", ""), *[repr(float(x)) if isinstance(x, float) else x for x in final]])


def _print_summary(res: BatchResult, log) -> None:
    rate = res.steps / res.wall_seconds if res.wall_seconds > 0 else float("inf")
    log(f"{'Finished' if res.finished else 'Stopped'} at t = {res.time:.6g} s after {res.steps} steps "
        f"in {res.wall_seconds:.3f} s ({rate:,.0f} steps/s)")
    for name, action, t in res.fired:
        log(f"Stop condition '{name}' became true at t = {t:.6g} s ({action})")
    prof = res.profile
    if prof.get("steps"):
        log("Profile (ms per step): " + ", ".join(f"{k.replace('_', ' ')} {prof[k]:.4f}"
                                                   for k in prof if k not in ("steps", "total"))
            + f"; total {prof['total']:.4f}")
    if res.error:
        log(f"ERROR: {res.error}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="psim-batch", description="Run a PSim project headless and export results.")
    p.add_argument("project")
    p.add_argument("--duration", type=float, help="simulated seconds (default: project setting)")
    p.add_argument("--dt", type=float, dest="delta_time", help="fixed time step Δt in seconds")
    p.add_argument("--integrator", help="semi_implicit_euler, explicit_euler, midpoint, velocity_verlet, rk4, dopri5")
    p.add_argument("--record-every", type=int, help="record every N steps")
    p.add_argument("--global", dest="globals_", action="append", default=[], metavar="NAME=VALUE",
                   help="override a global parameter (repeatable)")
    p.add_argument("--set", dest="props", action="append", default=[], metavar="OBJECT.PATH=VALUE",
                   help="override an object property, e.g. Ball.linear_velocity=[1,0,2] (repeatable)")
    p.add_argument("--var", dest="variables", action="append", default=[], metavar="OBJECT:QUANTITY[.C][@FRAME]",
                   help="CSV column(s), e.g. Ball:position, Ball:speed, Arm:angular_velocity.y (repeatable)")
    p.add_argument("--csv", dest="csv_path", help="write the selected variables (display units) as CSV")
    p.add_argument("--json", dest="json_path", help="write the full recording (SI) as JSON")
    p.add_argument("--report", dest="report_path", help="write the HTML simulation report")
    p.add_argument("--save-state", dest="state_out", help="write the final simulation state (.psimstate)")
    p.add_argument("--restore-state", dest="state_in", help="continue from a saved simulation state")
    p.add_argument("--max-wall", type=float, help="stop after this many wall-clock seconds")
    p.add_argument("--summary-json", help="write the run summary (steps, profile, conditions) as JSON")
    p.add_argument("--sweep", dest="sweeps", action="append", default=[], metavar="NAME=VALUES",
                   help="run once per value: a global parameter or OBJECT.PATH, values START:STOP:COUNT, a comma "
                        "list or a JSON list (repeat for a grid)")
    p.add_argument("--sweep-csv", help="write one row per sweep run (values, outcome, final --var values)")
    p.add_argument("--jobs", type=int, default=1, help="parallel processes for a sweep (default 1)")
    p.add_argument("--quiet", action="store_true", help="print errors only")
    args = p.parse_args(argv)
    log = (lambda *_a, **_k: None) if args.quiet else print
    if args.sweeps:
        return _main_sweep(args, log)
    if args.sweep_csv or args.jobs != 1:
        print("psim-batch: --sweep-csv and --jobs need --sweep", file=sys.stderr)
        return EXIT_INPUT
    try:
        res = run_batch(args.project, duration=args.duration, delta_time=args.delta_time,
                        integrator=args.integrator, record_every=args.record_every, globals_=args.globals_,
                        props=args.props, variables=args.variables, csv_path=args.csv_path,
                        json_path=args.json_path, report_path=args.report_path, state_out=args.state_out,
                        state_in=args.state_in, max_wall=args.max_wall, log=log)
    except BatchError as exc:
        print(f"psim-batch: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except KeyboardInterrupt:
        print("psim-batch: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    for level, msg in res.diagnostics:
        if level in ("error", "warning"):
            print(f"{level}: {msg}", file=sys.stderr)
    _print_summary(res, log if not res.error else (lambda m: print(m, file=sys.stderr)))
    if args.summary_json:
        from .project.serialization import write_json_atomic
        write_json_atomic(args.summary_json, res.to_dict())
    return EXIT_FAILED if res.error else EXIT_OK


def _main_sweep(args, log) -> int:
    if args.jobs < 1:
        print("psim-batch: --jobs must be at least 1", file=sys.stderr)
        return EXIT_INPUT
    kwargs = dict(duration=args.duration, delta_time=args.delta_time, integrator=args.integrator,
                  record_every=args.record_every, globals_=args.globals_, props=args.props, variables=args.variables,
                  csv_path=args.csv_path, json_path=args.json_path, report_path=args.report_path,
                  state_out=args.state_out, state_in=args.state_in, max_wall=args.max_wall)
    try:
        points, rows = run_sweep(args.project, args.sweeps, max(args.jobs, 1), args.sweep_csv, log, **kwargs)
    except BatchError as exc:
        print(f"psim-batch: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except KeyboardInterrupt:
        print("psim-batch: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    if not args.sweep_csv:                      # no file: print the table
        finals = list(dict.fromkeys(h for r in rows for h in r.get("final", {})))
        log("\t".join(["run", *points[0], "time [s]", *finals, "error"]))
        for k, (point, row) in enumerate(zip(points, rows), 1):
            log("\t".join([str(k), *(json.dumps(v) for v in point.values()), f"{row.get('time', 0.0):.6g}",
                           *(f"{row.get('final', {}).get(h, float('nan')):.6g}" for h in finals),
                           row.get("error", "")]))
    if args.summary_json:
        from .project.serialization import write_json_atomic
        write_json_atomic(args.summary_json, {"runs": [dict(row, values=point) for point, row in zip(points, rows)]})
    for k, row in enumerate(rows, 1):
        if row.get("error"):
            print(f"run {k}: {row['error']}", file=sys.stderr)
    return max((row["status"] for row in rows), default=EXIT_OK)


if __name__ == "__main__":
    sys.exit(main())
