"""CSV / JSON export of recorded data and the self-contained HTML simulation report."""
from __future__ import annotations

import base64
import csv
import datetime as _dt
import html
import json
import math
import os

import numpy as np

from .. import __version__
from ..units.unit_registry import UnitRegistry
from .analysis import STAT_COLUMNS, clean_row, statistics
from .recording import Recording
from .variables import Evaluator, Variable, info, label, recorded_objects


def _unit(units: UnitRegistry | None, quantity: str | None) -> str | None:
    if units is None or not quantity:
        return None
    return units.preferred_unit(quantity)


def series_table(ev: Evaluator, variables: list[Variable], units: UnitRegistry | None = None,
                 names: dict | None = None, frame_names: dict | None = None,
                 custom_quantities: dict | None = None) -> tuple[list[str], list[np.ndarray]]:
    """Column headers and value arrays (time first). Values in display units when `units` is given, else SI."""
    headers, cols = [], []
    for var in [Variable.time()] + [v for v in variables if v.quantity != "time"]:
        qi = info(var, custom_quantities)
        unit = _unit(units, qi.unit_quantity)
        values = ev.values(var)
        if unit:
            values = np.asarray(units.from_si(values, unit), float)
        elif qi.unit_quantity and units is None:
            unit = UnitRegistry().quantity(qi.unit_quantity).si_unit
        name = "time" if var.quantity == "time" else label(var, names, frame_names, custom_quantities)
        headers.append(f"{name} [{unit}]" if unit and unit != "1" else name)
        cols.append(values)
    return headers, cols


def export_csv(path: str, ev: Evaluator, variables: list[Variable], units: UnitRegistry | None = None,
               names: dict | None = None, frame_names: dict | None = None,
               custom_quantities: dict | None = None) -> int:
    headers, cols = series_table(ev, variables, units, names, frame_names, custom_quantities)
    n = len(cols[0]) if cols else 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for i in range(n):
            w.writerow(["" if not math.isfinite(c[i]) else repr(float(c[i])) for c in cols])
    return n


def _clean(a: np.ndarray):
    """JSON-safe nested lists (NaN → null)."""
    return [None if not math.isfinite(x) else float(x) for x in a] if a.ndim == 1 else [_clean(r) for r in a]


def export_json(path: str, rec: Recording, names: dict | None = None) -> None:
    """All recorded channels (SI) with metadata and the event log."""
    channels = {}
    for key in rec.keys():
        col = rec.column(key)
        channels["/".join(str(k) for k in key)] = _clean(col[:, 0] if col.shape[1] == 1 else col)
    payload = {
        "format": "psim-recording", "format_version": 1, "app_version": __version__,
        "exported": _dt.datetime.now().isoformat(timespec="seconds"),
        "units": "SI (m, kg, s, rad); orientations as quaternions (w, x, y, z)",
        "settings": rec.meta.get("settings", {}),
        "objects": {oid: {"name": (names or rec.meta.get("names", {})).get(oid, oid), **m}
                    for oid, m in rec.meta.get("objects", {}).items()},
        "processes": rec.meta.get("processes", {}),
        "recording": {"samples": rec.n, "interval_steps": rec.effective_interval, "decimations": rec.decimations,
                      "dropped_samples": rec.dropped, "stopped_at_limit": rec.full_stop},
        "time": _clean(rec.times()), "step": [int(s) for s in rec.steps()],
        "channels": channels,
        "events": [e.to_dict() for e in rec.event_list()],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)


def export_trajectory_csv(path: str, ev: Evaluator, object_id: str, frame: str | None = None,
                          point: str = "position", units: UnitRegistry | None = None) -> int:
    var = [Variable(object_id, point, c, frame) for c in "xyz"]
    return export_csv(path, ev, var, units, ev.rec.meta.get("names"))


# ---- report ------------------------------------------------------------------------------------------------------

_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px auto;max-width:1100px;color:#1d2025;
     padding:0 16px;line-height:1.45}
h1{font-size:1.6em;margin-bottom:0}h2{font-size:1.2em;margin-top:2em;border-bottom:1px solid #d6d9de;padding-bottom:4px}
.muted{color:#68707c}table{border-collapse:collapse;width:100%;font-size:.9em;margin:8px 0}
th,td{border:1px solid #d6d9de;padding:4px 8px;text-align:left;vertical-align:top}th{background:#f2f4f7}
td.num{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,Consolas,monospace}
img{max-width:100%;border:1px solid #d6d9de;margin:8px 0}
@media (prefers-color-scheme:dark){body{background:#17191d;color:#e3e6ea}th{background:#22262c}
th,td,img,h2{border-color:#353a42}.muted{color:#9aa1ab}}
"""


def _fmt(x: float) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{x:.6g}"


def generate_report(path: str, rec: Recording, scene=None, units: UnitRegistry | None = None,
                    title: str = "Simulation report", images: list[tuple[str, bytes]] | None = None) -> None:
    """Write a self-contained HTML report: settings, objects, processes, statistics, events, plots."""
    units = units or UnitRegistry()
    ev = Evaluator(rec, scene)
    names = dict(rec.meta.get("names", {}))
    if scene is not None:
        names.update({o.id: o.name for o in scene.objects.values()})
    e = html.escape
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
             f"content='width=device-width,initial-scale=1'><title>{e(title)}</title><style>{_CSS}</style></head><body>",
             f"<h1>{e(title)}</h1><p class='muted'>Generated {_dt.datetime.now():%Y-%m-%d %H:%M} by PSim Studio "
             f"{__version__}. Values from recorded samples; units as configured in the project.</p>"]
    s = rec.meta.get("settings", {})
    rows = [("Δt", f"{s.get('delta_time', '—')} s"), ("Duration", f"{s.get('duration', '—')} s"),
            ("Integrator", str(s.get("integrator", "—")).replace("_", " ")),
            ("Update rate", f"{s.get('update_rate_hz', '—')} Hz ({s.get('pacing', '')})"),
            ("Recording", rec.status_text())]
    if rec.n:
        rows.insert(2, ("Recorded interval", f"{rec.times()[0]:.6g} s … {rec.times()[-1]:.6g} s"))
    parts.append("<h2>Simulation settings</h2><table>" + "".join(f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>"
                                                                  for k, v in rows) + "</table>")
    objs = recorded_objects(rec)
    mu = units.preferred_unit("mass")
    parts.append("<h2>Recorded objects</h2><table><tr><th>Name</th><th>ID</th><th>Type</th>"
                 f"<th>Mass [{e(mu)}]</th><th>Simulated</th></tr>")
    for oid in objs:
        m = rec.meta.get("objects", {}).get(oid, {})
        parts.append(f"<tr><td>{e(names.get(oid, oid))}</td><td>{e(oid)}</td>"
                     f"<td>{e(rec.meta.get('types', {}).get(oid, ''))}</td>"
                     f"<td class='num'>{_fmt(units.from_si(m.get('mass', float('nan')), mu))}</td>"
                     f"<td>{'body' if m.get('body') else 'carried by an ancestor'}</td></tr>")
    parts.append("</table>")
    procs = rec.meta.get("processes", {})
    if procs:
        parts.append("<h2>Processes</h2><table><tr><th>Assignment</th><th>Process</th><th>Kind</th></tr>" +
                     "".join(f"<tr><td>{e(uid)}</td><td>{e(p['name'])} <span class='muted'>({e(p['process_id'])})"
                             f"</span></td><td>{e(p['kind'])}</td></tr>" for uid, p in procs.items()) + "</table>")
    if rec.n and objs:
        t = ev.values(Variable.time())
        parts.append("<h2>Statistics</h2><p class='muted'>Time-weighted over the recorded interval.</p>"
                     "<table><tr><th>Object</th><th>Quantity</th><th>Unit</th>" +
                     "".join(f"<th>{c}</th>" for c in STAT_COLUMNS) + "</tr>")
        for oid in objs:
            for var in (Variable(oid, "position", "x"), Variable(oid, "position", "y"), Variable(oid, "position", "z"),
                        Variable(oid, "speed"), Variable(oid, "acceleration", "mag"),
                        Variable(oid, "angular_velocity", "mag"), Variable(oid, "kinetic_energy")):
                qi = info(var)
                unit = units.preferred_unit(qi.unit_quantity)
                st = statistics(t, np.asarray(units.from_si(ev.values(var), unit), float))
                row = clean_row(st.as_row()) if st else [math.nan] * 7
                cells = "".join(f"<td class='num'>{_fmt(x)}</td>" for x in row)
                generic = label(Variable(None, var.quantity, var.component, var.frame))
                parts.append(f"<tr><td>{e(names.get(oid, oid))}</td><td>{e(generic)}</td><td>{e(unit)}</td>"
                             f"{cells}</tr>")
        parts.append("</table>")
    for caption, png in images or []:
        parts.append(f"<h2>{e(caption)}</h2><img alt='{e(caption)}' src='data:image/png;base64,"
                     f"{base64.b64encode(png).decode()}'>")
    events = rec.event_list()
    parts.append("<h2>Event log</h2>")
    if events:
        parts.append("<table><tr><th>t [s]</th><th>Step</th><th>Event</th><th>Message</th></tr>" +
                     "".join(f"<tr><td class='num'>{x.time:.6g}</td><td class='num'>{x.step}</td><td>{e(x.kind)}</td>"
                             f"<td>{e(x.message)}</td></tr>" for x in events) + "</table>")
    else:
        parts.append("<p class='muted'>No events.</p>")
    parts.append("</body></html>")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    os.replace(tmp, path)
