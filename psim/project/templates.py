"""Built-in project templates (File ▸ New from Template)."""
from __future__ import annotations

import math

from ..scene.editing import SceneEditor
from ..scene.scene_manager import Scene


def _primitive_gallery(scene: Scene, editor: SceneEditor) -> None:
    types = ["basic.cube", "basic.cuboid", "basic.sphere", "basic.ellipsoid", "basic.cylinder", "basic.cone",
             "basic.frustum", "basic.torus", "basic.capsule", "basic.pyramid", "basic.prism", "basic.hemisphere",
             "basic.disk", "basic.plane"]
    objs = []
    cols = 5
    for k, t in enumerate(types):
        x = (k % cols - (cols - 1) / 2) * 2.0
        y = (k // cols - 1) * 2.2
        o = scene.create_object(t, position=[x, y, 0.0])
        lo, _ = o.local_bounds()
        o.position[2] = -float(lo[2]) + (0.001 if t in ("basic.disk", "basic.plane") else 0.0)
        o.overlays["name"] = True
        objs.append(o)
    editor.add_objects(objs, text="Primitive gallery", select=False)


def _mechanical_demo(scene: Scene, editor: SceneEditor) -> None:
    base = scene.create_object("basic.cuboid", name="Base Plate", params={"length": 4.0, "width": 2.5, "height": 0.1},
                               position=[0, 0, 0.05])
    base.color = [0.42, 0.44, 0.48]
    base.material, base.metallic, base.roughness = "Brushed Metal", 1.0, 0.45
    base.density = 7850.0
    frame = scene.create_object("frame.reference", name="Machine Frame", params={"axis_length": 0.6},
                                position=[-1.6, -0.9, 0.1])
    g1 = scene.create_object("mech.gear", name="Input Gear", params={"teeth": 24, "outer_radius": 0.6,
                                                                     "root_radius": 0.52, "thickness": 0.15},
                             position=[-0.6, 0.0, 0.9])
    g1.set_euler([math.pi / 2, 0, 0])
    g2 = scene.create_object("mech.gear", name="Output Gear", params={"teeth": 12, "outer_radius": 0.32,
                                                                      "root_radius": 0.26, "thickness": 0.15},
                             position=[0.33, 0.0, 0.9])
    g2.set_euler([math.pi / 2, 0, math.pi / 12])
    g2.color = [0.82, 0.62, 0.28]
    shaft1 = scene.create_object("basic.cylinder", name="Input Shaft", params={"radius": 0.05, "height": 1.0},
                                 position=[-0.6, 0.0, 0.9])
    shaft1.set_euler([math.pi / 2, 0, 0])
    shaft2 = scene.create_object("basic.cylinder", name="Output Shaft", params={"radius": 0.04, "height": 1.0},
                                 position=[0.33, 0.0, 0.9])
    shaft2.set_euler([math.pi / 2, 0, 0])
    for s in (shaft1, shaft2):
        s.color = [0.75, 0.76, 0.79]
        s.material, s.metallic, s.roughness = "Polished Metal", 1.0, 0.15
        s.density = 7850.0
    support = scene.create_object("mech.i_beam", name="Support Beam", params={"width": 0.2, "height": 0.25,
                                                                              "web": 0.02, "flange": 0.03, "length": 0.8},
                                  position=[1.5, 0.6, 0.5])
    spring = scene.create_object("mech.spring", name="Return Spring", params={"coil_radius": 0.12, "wire_radius": 0.015,
                                                                              "length": 0.6, "turns": 9},
                                 position=[1.5, -0.6, 0.4])
    mass = scene.create_object("body.point_mass", name="Probe Mass", position=[1.5, -0.6, 0.85])
    mass.overlays.update(name=True, id=True, mass=True)
    arrow = scene.create_object("measure.vector_arrow", name="Load Direction", params={"length": 0.8},
                                position=[1.5, -0.6, 1.0])
    arrow.set_euler([0, -math.pi / 2, 0])
    g1.overlays.update(name=True, com=True)
    g2.overlays.update(name=True)
    editor.add_objects([base, frame, g1, g2, shaft1, shaft2, support, spring, mass, arrow], text="Mechanical demo",
                       select=False)


PROJECT_TEMPLATES = {
    "empty": ("Empty Scene", None),
    "primitives": ("Primitive Gallery", _primitive_gallery),
    "mechanical": ("Mechanical Demo", _mechanical_demo),
}
