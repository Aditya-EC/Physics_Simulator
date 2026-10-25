"""Object library catalogue: built-in types, composite templates, saved templates, imported meshes.

Every library item can produce (a) a preview mesh for thumbnails and the
drag-and-drop ghost and (b) serialized object states that the SceneEditor
instantiates with fresh IDs.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

from ..project.autosave import data_dir
from ..scene import geometry as g
from ..scene.builtin_types import mesh_assets
from ..scene.object_registry import CATEGORIES, registry
from ..scene.scene_manager import Scene


@dataclass
class LibraryItem:
    key: str                     # unique, e.g. "type:basic.cube"
    name: str
    category: str
    description: str
    kind: str                    # type | composite | template | mesh
    type_id: str | None = None
    params: dict = field(default_factory=dict)
    states: list[dict] | None = None  # for composites / templates (root at origin)
    tags: tuple[str, ...] = ()


def _states_from(build) -> list[dict]:
    """Build a composite in a scratch scene and return states with the root at the origin."""
    s = Scene()
    build(s)
    return [o.to_dict() for o in sorted(s.objects.values(), key=lambda o: s.depth(o.id))]


def _composite_dumbbell(s: Scene) -> None:
    root = s.create_object("group.assembly", name="Dumbbell")
    s.add_objects([root])
    rod = s.create_object("basic.cylinder", name="Bar", params={"radius": 0.05, "height": 1.2})
    rod.set_euler([0, np.pi / 2, 0])
    rod.color = [0.72, 0.73, 0.76]
    rod.material, rod.metallic, rod.roughness = "Brushed Metal", 1.0, 0.45
    rod.density = 7850.0
    a = s.create_object("basic.sphere", name="Weight A", params={"radius": 0.22}, position=[-0.6, 0, 0])
    b = s.create_object("basic.sphere", name="Weight B", params={"radius": 0.22}, position=[0.6, 0, 0])
    for o in (rod, a, b):
        o.parent_id = root.id
    a.color = b.color = [0.22, 0.24, 0.27]
    a.density = b.density = 7200.0
    s.add_objects([rod, a, b])


def _composite_wheel_axle(s: Scene) -> None:
    root = s.create_object("group.assembly", name="Wheel & Axle")
    s.add_objects([root])
    axle = s.create_object("basic.cylinder", name="Axle", params={"radius": 0.04, "height": 1.4})
    axle.set_euler([np.pi / 2, 0, 0])
    axle.color = [0.70, 0.72, 0.75]
    parts = [axle]
    for k, y in enumerate((-0.6, 0.6)):
        w = s.create_object("basic.cylinder", name=f"Wheel {k + 1}", params={"radius": 0.35, "height": 0.12},
                            position=[0, y, 0])
        w.set_euler([np.pi / 2, 0, 0])
        w.color = [0.18, 0.19, 0.21]
        w.material, w.roughness = "Rubber", 0.95
        w.density = 1100.0
        parts.append(w)
    for o in parts:
        o.parent_id = root.id
    s.add_objects(parts)


def _composite_pendulum(s: Scene) -> None:
    root = s.create_object("frame.reference", name="Pendulum Pivot", params={"axis_length": 0.35})
    s.add_objects([root])
    rod = s.create_object("measure.line_segment", name="Rod", params={"length": 1.0, "radius": 0.01}, position=[0, 0, -0.5])
    rod.set_euler([0, np.pi / 2, 0])
    bob = s.create_object("basic.sphere", name="Bob", params={"radius": 0.12}, position=[0, 0, -1.0])
    bob.color = [0.85, 0.45, 0.25]
    bob.density = 7850.0
    for o in (rod, bob):
        o.parent_id = root.id
    s.add_objects([rod, bob])


def _composite_spring_mass(s: Scene) -> None:
    root = s.create_object("group.assembly", name="Spring-Mass")
    s.add_objects([root])
    base = s.create_object("basic.cuboid", name="Support", params={"length": 0.6, "width": 0.6, "height": 0.08},
                           position=[0, 0, 0.75])
    base.color = [0.45, 0.47, 0.52]
    spring = s.create_object("mech.spring", name="Spring", params={"coil_radius": 0.12, "wire_radius": 0.015,
                                                                    "length": 0.9, "turns": 10}, position=[0, 0, 0.26])
    mass = s.create_object("body.rigid_box", name="Mass", params={"length": 0.35, "width": 0.35, "height": 0.3},
                           position=[0, 0, -0.34])
    for o in (base, spring, mass):
        o.parent_id = root.id
    s.add_objects([base, spring, mass])


def _composite_gear_pair(s: Scene) -> None:
    root = s.create_object("group.assembly", name="Gear Pair")
    s.add_objects([root])
    g1 = s.create_object("mech.gear", name="Driver Gear", params={"teeth": 20, "outer_radius": 0.5,
                                                                  "root_radius": 0.43, "thickness": 0.12},
                         position=[-0.46, 0, 0])
    g2 = s.create_object("mech.gear", name="Driven Gear", params={"teeth": 12, "outer_radius": 0.32,
                                                                  "root_radius": 0.26, "thickness": 0.12},
                         position=[0.36, 0, 0])
    g2.set_euler([0, 0, np.pi / 12])
    g2.color = [0.80, 0.62, 0.30]
    for o in (g1, g2):
        o.parent_id = root.id
    s.add_objects([g1, g2])


COMPOSITES = [
    ("dumbbell", "Dumbbell", "Two spherical weights joined by a bar (assembly).", _composite_dumbbell),
    ("wheel_axle", "Wheel & Axle", "Axle with two wheels (assembly).", _composite_wheel_axle),
    ("pendulum", "Simple Pendulum", "Pivot frame, rod and bob arranged as a pendulum (assembly).", _composite_pendulum),
    ("spring_mass", "Spring-Mass", "Support, coil spring and mass block (assembly).", _composite_spring_mass),
    ("gear_pair", "Gear Pair", "Two meshing spur gears (assembly).", _composite_gear_pair),
]


def templates_dir() -> str:
    d = os.path.join(data_dir(), "templates")
    os.makedirs(d, exist_ok=True)
    return d


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9 _.-]+", "_", name).strip() or "template"


class Library:
    def __init__(self) -> None:
        self._items: dict[str, LibraryItem] = {}
        self._mesh_cache: dict[str, g.Mesh] = {}
        self.errors: list[str] = []
        self.reload()

    def reload(self) -> None:
        self._items.clear()
        self._mesh_cache.clear()
        for t in registry().types():
            if t.user_creatable:
                self._add(LibraryItem(f"type:{t.type_id}", t.name, t.category, t.description, "type", t.type_id,
                                      tags=t.tags))
        for key, name, desc, build in COMPOSITES:
            self._add(LibraryItem(f"composite:{key}", name, "Composite Shapes", desc, "composite",
                                  states=_states_from(build)))
        self.reload_templates()
        self.reload_meshes()

    def reload_templates(self) -> None:
        for k in [k for k, v in self._items.items() if v.kind == "template"]:
            del self._items[k]
            self._mesh_cache.pop(k, None)
        self.errors = []
        for fn in sorted(os.listdir(templates_dir())):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(templates_dir(), fn)
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                states = data["objects"]
                if not isinstance(states, list) or not states:
                    raise ValueError("no objects")
                for st in states:
                    registry().get(st["type"])
                self._add(LibraryItem(f"template:{fn}", str(data.get("name", fn[:-5])), "Saved Templates",
                                      str(data.get("description", "Saved object template")), "template",
                                      states=states))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.errors.append(f"Template '{fn}' could not be loaded: {exc}")

    def reload_meshes(self) -> None:
        for k in [k for k, v in self._items.items() if v.kind == "mesh"]:
            del self._items[k]
            self._mesh_cache.pop(k, None)
        for key in mesh_assets.keys():
            self._add(LibraryItem(f"mesh:{key}", mesh_assets.name(key), "User-Defined Objects",
                                  "Imported polygonal mesh", "mesh", "user.mesh", params={"asset": key}))

    def _add(self, item: LibraryItem) -> None:
        self._items[item.key] = item

    # ---- queries ---------------------------------------------------------------------------------
    def items(self) -> list[LibraryItem]:
        return list(self._items.values())

    def get(self, key: str) -> LibraryItem | None:
        return self._items.get(key)

    def categories(self) -> list[str]:
        return list(CATEGORIES)

    def search(self, text: str) -> list[LibraryItem]:
        words = text.lower().split()
        out = []
        for it in self._items.values():
            hay = f"{it.name} {it.category} {it.description} {' '.join(it.tags)}".lower()
            if all(w in hay for w in words):
                out.append(it)
        return out

    # ---- geometry ----------------------------------------------------------------------------------
    def preview_mesh(self, key: str) -> g.Mesh | None:
        """Combined mesh of the item in its own frame (used for thumbnails and drop previews)."""
        if key in self._mesh_cache:
            return self._mesh_cache[key]
        it = self._items.get(key)
        if it is None:
            return None
        if it.kind in ("type", "mesh"):
            td = registry().get(it.type_id)
            mesh = td.build_mesh(td.coerce_params(it.params))
        else:
            s = Scene()
            for st in it.states:
                from ..scene.object_base import SceneObject
                obj = SceneObject.from_dict(st)
                s.objects[obj.id] = obj
            parts = []
            for o in s.objects.values():
                m = o.mesh()
                parts.append(m.transformed(s.world_matrix(o.id)).with_color(o.color) if not o.typedef.uses_vertex_colors
                             else m.transformed(s.world_matrix(o.id)))
            mesh = g.merge(parts, solid=False)
        self._mesh_cache[key] = mesh
        return mesh

    def thumbnail_spec(self, key: str) -> tuple[g.Mesh | None, tuple[float, float, float], bool, tuple[float, float]]:
        it = self._items[key]
        if it.kind == "type":
            td = registry().get(it.type_id)
            params = td.default_params()
            if "segments" in params:
                params["segments"] = 32
            mesh = td.build_mesh(params)
            from ..scene.materials import PALETTE
            color = td.default_color or PALETTE[sum(map(ord, td.type_id)) % len(PALETTE)]
            return mesh, color, td.uses_vertex_colors, td.thumbnail_view
        mesh = self.preview_mesh(key)
        return mesh, (0.62, 0.66, 0.72), it.kind in ("composite", "template"), (35.0, 25.0)

    # ---- templates ----------------------------------------------------------------------------------
    def save_template(self, name: str, states: list[dict], description: str = "") -> str:
        """Store objects as a reusable template; positions are made relative to the first root object."""
        if not states:
            raise ValueError("Nothing to save")
        ids = {s["id"] for s in states}
        roots = [s for s in states if s.get("parent") not in ids]
        origin = np.asarray(roots[0].get("position", [0, 0, 0]), float)
        out = []
        for s in states:
            s = json.loads(json.dumps(s))
            if s["id"] not in {r["id"] for r in roots}:
                out.append(s)
                continue
            s["parent"] = None
            s["position"] = list(np.asarray(s["position"], float) - origin)
            out.append(s)
        path = os.path.join(templates_dir(), _safe_filename(name) + ".json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"format": "psim-template", "version": 1, "name": name, "description": description,
                       "objects": out}, fh, indent=1)
        self.reload_templates()
        return path

    def delete_template(self, key: str) -> None:
        it = self._items.get(key)
        if it is None or it.kind != "template":
            return
        os.remove(os.path.join(templates_dir(), key.split(":", 1)[1]))
        self.reload_templates()

