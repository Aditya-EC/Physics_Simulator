"""Versioned project file format (.psim, JSON).

Top-level layout (format version 2; version 1 files — without process
assignments — are still read):

    {
      "format": "psim-project",
      "format_version": 1,
      "app_version": "0.1.0",
      "metadata":   {"name", "description", "author", "created", "modified"},
      "id_counter": <next numeric object id>,
      "objects":    [<SceneObject.to_dict()>...],   # scene order
      "processes":  [<ProcessAssignment.to_dict()>...],  # evaluation order (v2)
      "process_counter": <next numeric process assignment id> (v2)
      "global_parameters": [{"name", "type", "quantity", "value", "description"}...] (v2, optional)
      "reference_frames": [{"id", "name", "parent", "position", "orientation", "follow_rotation", …}...],
      "frame_counter": next FRM- number (v2, optional; frames are never reused)
      "joints": [<Joint.to_dict()>...], "joint_counter": next JNT- number (v2, optional)
      "measurements": [{"id", "name", "kind", "points", "visible"}...], "measurement_counter" (v2, optional)
      "assets":     {"meshes": {<asset id>: {"name", "vertices", "indices"}}},
      "units":      {"preferred": {...}, "custom_units": [...]},
      "view":       <ViewSettings.to_dict()>,
      "simulation": <SimulationSettings.to_dict()>,
      "ui":         {"layout": <base64 Qt window state>}   (optional)
      "plots":      [<PlotSpec.to_dict()>...]              (optional; recorded data is not stored)
    }

Project files are pure data. Loading never evaluates code: every field is
type-checked, unknown object types are rejected, and values are validated
through the property system.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
import time
from dataclasses import dataclass, field

from .. import __version__
from ..core.ids import DuplicateIdError
from ..properties.property_system import PropertyError
from ..rendering.view_settings import ViewSettings
from ..scene.builtin_types import mesh_assets
from ..scene.object_base import SceneObject
from ..scene.object_registry import RegistryError
from ..scene.scene_manager import Scene
from ..simulation.settings import SimulationSettings
from ..units.unit_registry import UnitRegistry

FORMAT = "psim-project"
FORMAT_VERSION = 2
MAX_FILE_BYTES = 512 * 1024 * 1024


class ProjectError(Exception):
    pass


@dataclass
class ProjectData:
    metadata: dict = field(default_factory=dict)
    view: ViewSettings = field(default_factory=ViewSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    ui: dict = field(default_factory=dict)
    plots: list[dict] = field(default_factory=list)   # PlotSpec dicts of the plot workspace
    warnings: list[str] = field(default_factory=list)


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def project_to_dict(scene: Scene, units: UnitRegistry, data: ProjectData) -> dict:
    ordered = list(scene.objects.values())  # parent links are resolved after loading, so order is preserved
    used_assets = {o.params.get("asset") for o in ordered if o.type_id == "user.mesh"}
    meta = dict(data.metadata)
    meta.setdefault("created", now_iso())
    meta["modified"] = now_iso()
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "app_version": __version__,
        "metadata": meta,
        "id_counter": scene.ids.next_value,
        "objects": [o.to_dict() for o in ordered],
        "processes": scene.process_snapshot(),
        "process_counter": scene.next_process_number,
        "global_parameters": [dict(g) for g in scene.global_parameters],
        "reference_frames": scene.frames_snapshot(),
        "frame_counter": scene.next_frame_number,
        "joints": scene.joints_snapshot(),
        "joint_counter": scene.next_joint_number,
        "measurements": scene.measurements_snapshot(),
        "measurement_counter": scene.next_measurement_number,
        "assets": {"meshes": mesh_assets.to_dict({a for a in used_assets if a})},
        "units": units.to_dict(),
        "view": data.view.to_dict(),
        "simulation": data.simulation.to_dict(),
        "ui": dict(data.ui),
        "plots": [dict(p) for p in data.plots],
    }


def save_project(path: str, scene: Scene, units: UnitRegistry, data: ProjectData) -> None:
    payload = project_to_dict(scene, units, data)
    data.metadata = payload["metadata"]
    write_json_atomic(path, payload)


REPLACE_ATTEMPTS = 10           # × REPLACE_WAIT seconds: Windows refuses to replace a file another process holds
REPLACE_WAIT = 0.05


def _replace(src: str, dst: str) -> None:
    """os.replace, retried briefly on PermissionError: on Windows a virus scanner, search indexer or sync client
    that has just opened the target makes the replace fail for a moment (POSIX never refuses it)."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_WAIT)


def write_json_atomic(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".psim-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        # mkstemp creates 0600 files; keep the existing file's mode, else use the default (umask) mode
        try:
            mode = os.stat(path).st_mode & 0o777
        except FileNotFoundError:
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o666 & ~umask
        os.chmod(tmp, mode)
        _replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_project_file(path: str) -> dict:
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            raise ProjectError("Project file is too large")
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        raise ProjectError(f"File not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ProjectError(f"Not a valid project file (JSON error at line {exc.lineno}: {exc.msg})") from None
    except UnicodeDecodeError:
        raise ProjectError("Not a valid project file (not UTF-8 text)") from None
    return raw


def load_project_dict(raw: dict, scene: Scene, units: UnitRegistry) -> ProjectData:
    """Validate `raw` and load it into `scene` / `units`. The scene is only modified on success."""
    if not isinstance(raw, dict) or raw.get("format") != FORMAT:
        raise ProjectError("Not a PSim project file")
    version = raw.get("format_version")
    if not isinstance(version, int):
        raise ProjectError("Project file has no format version")
    if version > FORMAT_VERSION:
        raise ProjectError(f"Unsupported project version {version} (this application reads up to {FORMAT_VERSION}). "
                           "Please update the application.")
    data = ProjectData()
    # assets first so user meshes resolve
    assets = raw.get("assets", {})
    if isinstance(assets, dict) and isinstance(assets.get("meshes"), dict):
        data.warnings += mesh_assets.load_dict(assets["meshes"])

    objects: list[SceneObject] = []
    seen: set[str] = set()
    for i, od in enumerate(raw.get("objects", [])):
        if not isinstance(od, dict):
            raise ProjectError(f"Object #{i} is malformed")
        try:
            obj = SceneObject.from_dict(od)
        except (PropertyError, RegistryError) as exc:
            raise ProjectError(str(exc)) from None
        if obj.id in seen:
            raise ProjectError(f"Duplicate object ID '{obj.id}'")
        seen.add(obj.id)
        if obj.type_id == "user.mesh" and mesh_assets.get(obj.params.get("asset", "")) is None:
            data.warnings.append(f"{obj.id}: missing mesh asset '{obj.params.get('asset')}' (shown as a placeholder box)")
        objects.append(obj)
    from ..processes.process_base import ProcessAssignment, ProcessError
    from ..processes.process_registry import process_registry
    assignments = []
    for i, ad in enumerate(raw.get("processes", []) if version >= 2 else []):
        try:
            if not isinstance(ad, dict):
                raise ProcessError("malformed entry")
            a = ProcessAssignment.from_dict(ad)
            if any(a.uid == b.uid for b in assignments):
                raise ProcessError(f"duplicate assignment id '{a.uid}'")
        except ProcessError as exc:
            raise ProjectError(f"Process assignment #{i}: {exc}") from None
        if not process_registry().has(a.process_id):
            data.warnings.append(f"{a.uid}: process '{a.process_id}' is not available in this installation "
                                 "(kept but disabled)")
            a.enabled = False
        assignments.append(a)
    from ..processes.global_parameters import validate_globals
    try:
        globals_ = validate_globals(raw.get("global_parameters", []) if isinstance(raw.get("global_parameters", []), list)
                                    else [])
    except ProcessError as exc:
        raise ProjectError(str(exc)) from None
    from ..coordinates.frames import FRAME_ID, FrameError, validate_frames
    try:
        frames = validate_frames(raw.get("reference_frames", []))
    except FrameError as exc:
        raise ProjectError(str(exc)) from None
    for f in frames:
        if f.parent and f.parent not in seen:
            data.warnings.append(f"{f.id} ({f.name}): attached to missing object '{f.parent}'")
    from ..physics.joints import JOINT_ID, JointError, validate_joints
    try:
        joints = validate_joints(raw.get("joints", []))
    except JointError as exc:
        raise ProjectError(str(exc)) from None
    for j in joints:
        for b in (j.body_a, j.body_b):
            if b and b not in seen:
                data.warnings.append(f"{j.id} ({j.name}): connects the missing object '{b}'")
    from ..scene.measurements import MEASUREMENT_ID, MeasurementError, validate_measurements
    try:
        measurements = validate_measurements(raw.get("measurements", []))
    except MeasurementError as exc:
        raise ProjectError(str(exc)) from None
    kept = []
    for m in measurements:
        missing = [o for o in m.objects if o not in seen]
        if missing:
            data.warnings.append(f"{m.id} ({m.name}): measured the missing object '{missing[0]}'; removed")
        else:
            kept.append(m)
    for obj in objects:
        if obj.parent_id and obj.parent_id not in seen:
            data.warnings.append(f"{obj.id}: parent '{obj.parent_id}' not found; object moved to the scene root")
            obj.parent_id = None

    scene.blockSignals(True)
    try:
        scene.assignments = assignments
        scene.global_parameters = globals_
        scene.frames = frames
        try:
            scene.next_frame_number = max(int(raw.get("frame_counter", 1)), 1,
                                          *(int(FRAME_ID.match(f.id).group(1)) + 1 for f in frames))
        except (TypeError, ValueError):
            scene.next_frame_number = 1 + max((int(FRAME_ID.match(f.id).group(1)) for f in frames), default=0)
        scene.joints = joints
        try:
            scene.next_joint_number = max(int(raw.get("joint_counter", 1)), 1,
                                          *(int(JOINT_ID.match(j.id).group(1)) + 1 for j in joints))
        except (TypeError, ValueError):
            scene.next_joint_number = 1 + max((int(JOINT_ID.match(j.id).group(1)) for j in joints), default=0)
        scene.measurements = kept
        try:
            scene.next_measurement_number = max(int(raw.get("measurement_counter", 1)), 1,
                                                *(int(MEASUREMENT_ID.match(m.id).group(1)) + 1 for m in measurements))
        except (TypeError, ValueError):
            scene.next_measurement_number = 1 + max((int(MEASUREMENT_ID.match(m.id).group(1))
                                                     for m in measurements), default=0)
        scene.simulating = False
        try:
            scene.next_process_number = max(int(raw.get("process_counter", 1)), 1)
        except (TypeError, ValueError):
            scene.next_process_number = 1
        scene.runtime = {}
        scene.objects.clear()
        scene.set_selection([])
        try:
            scene.ids.reset(int(raw.get("id_counter", 1)))
        except (TypeError, ValueError):
            scene.ids.reset(1)
        for obj in objects:
            scene.ids.reserve(obj.id)
            scene.objects[obj.id] = obj
    except DuplicateIdError as exc:
        raise ProjectError(str(exc)) from None
    finally:
        scene.blockSignals(False)
    cycles = [p for p in scene.validate() if "cycle" in p]
    if cycles:
        for obj in objects:
            if obj.id in scene.ancestors(obj.id):
                obj.parent_id = None
        data.warnings += cycles
    scene.sceneReset.emit()
    scene.processesChanged.emit()
    scene.framesChanged.emit()
    scene.jointsChanged.emit()

    if isinstance(raw.get("units"), dict):
        data.warnings += units.load_dict(raw["units"])
    if isinstance(raw.get("view"), dict):
        data.view = ViewSettings.from_dict(raw["view"])
    if isinstance(raw.get("simulation"), dict):
        data.simulation = SimulationSettings.from_dict(raw["simulation"])
        data.warnings += [f"Simulation settings: {e}" for e in data.simulation.validate()]
    if isinstance(raw.get("metadata"), dict):
        data.metadata = {str(k): str(v) for k, v in raw["metadata"].items()}
    if isinstance(raw.get("plots"), list):
        from ..recording.plot_spec import PlotSpec, PlotSpecError
        for i, p in enumerate(raw["plots"]):
            try:
                data.plots.append(PlotSpec.from_dict(p).to_dict())
            except PlotSpecError as exc:
                data.warnings.append(f"Plot #{i + 1} skipped: {exc}")
    if isinstance(raw.get("ui"), dict):
        data.ui = {k: v for k, v in raw["ui"].items() if isinstance(v, str)}
    return data


def load_project(path: str, scene: Scene, units: UnitRegistry) -> ProjectData:
    return load_project_dict(read_project_file(path), scene, units)

