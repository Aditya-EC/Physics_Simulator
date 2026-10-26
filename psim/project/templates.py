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


def _simulation_demo(scene: Scene, editor: SceneEditor) -> None:
    from ..processes.process_base import TargetFilter
    ground = scene.create_object("basic.plane", name="Ground", params={"length": 12, "width": 8})
    ground.color = [0.32, 0.34, 0.38]
    ball = scene.create_object("basic.sphere", name="Projectile", params={"radius": 0.15}, position=[-5, -2, 0.15])
    ball.linear_velocity = [4.0, 0.0, 6.5]
    ball.color = [0.92, 0.45, 0.20]
    ball.density = 7850.0
    ball.overlays.update(name=True, velocity=True, acceleration=True)
    anchor = scene.create_object("body.point_mass", name="Driven Anchor", position=[2.5, 2.0, 3.0])
    block = scene.create_object("body.rigid_box", name="Hanging Block", params={"length": 0.4, "width": 0.4, "height": 0.4},
                                position=[2.5, 2.0, 1.6])
    block.mass_mode, block.mass_input = "mass", 2.0
    block.overlays.update(name=True, position=True)
    gear = scene.create_object("mech.gear", name="Spinning Gear", params={"teeth": 18, "outer_radius": 0.6,
                                                                         "root_radius": 0.52, "thickness": 0.15},
                               position=[0.0, 2.5, 1.0])
    gear.set_euler([math.pi / 2, 0, 0])
    marker = scene.create_object("frame.marker", name="Orbiting Marker", position=[3.0, -2.0, 0.6])
    puck = scene.create_object("basic.cylinder", name="Pushed Puck", params={"radius": 0.3, "height": 0.15},
                               position=[-2.0, 2.0, 0.075])
    puck.force = [6.0, 0.0, 0.0]
    puck.torque = [0.0, 0.0, 0.6]
    puck.overlays.update(name=True, velocity=True)
    editor.add_objects([ground, ball, anchor, block, gear, marker, puck], text="Simulation demo", select=False)
    editor.assign_process("dyn.gravity", TargetFilter(ids=[ball.id, block.id]), label="Earth gravity")
    editor.assign_process("kin.oscillation", TargetFilter(ids=[anchor.id]),
                          {"direction": [0, 0, 1], "amplitude": 0.3, "frequency": 0.6})
    editor.assign_process("dyn.spring", TargetFilter(ids=[anchor.id, block.id]),
                          {"stiffness": 60.0, "rest_length": 1.0, "damping": 0.8})
    editor.assign_process("rot.local_axis", TargetFilter(ids=[gear.id]), {"axis": [0, 0, 1], "angular_speed": 1.5})
    editor.assign_process("kin.circular_motion", TargetFilter(ids=[marker.id]),
                          {"center": [3.0, -2.0, 0.6], "radius": 1.0, "angular_speed": 1.2})
    editor.assign_process("dyn.applied_force", TargetFilter(ids=[puck.id]))
    editor.assign_process("dyn.damping", TargetFilter(ids=[puck.id]), {"linear": 2.5, "angular": 0.4})


def _formula_demo(scene: Scene, editor: SceneEditor) -> None:
    """Formula-defined library processes: a driven tether in the wind, discrete update rules, a custom output."""
    from ..processes.process_base import TargetFilter
    from ..properties.custom_properties import CustomProperty
    ground = scene.create_object("basic.plane", name="Ground", params={"length": 12, "width": 8})
    ground.color = [0.32, 0.34, 0.38]
    anchor = scene.create_object("body.point_mass", name="Lissajous Anchor", position=[0.0, 0.0, 2.6])
    anchor.overlays.update(name=True)
    bob = scene.create_object("basic.sphere", name="Tethered Bob", params={"radius": 0.12}, position=[0.0, 0.0, 1.4])
    bob.mass_mode, bob.mass_input = "mass", 0.5
    bob.color = [0.30, 0.62, 0.95]
    bob.overlays.update(name=True, velocity=True)
    ball = scene.create_object("basic.sphere", name="Monitored Ball", params={"radius": 0.1}, position=[-4.5, -2.0, 0.1])
    ball.mass_mode, ball.mass_input = "mass", 0.2
    ball.linear_velocity = [3.0, 0.0, 7.0]
    ball.color = [0.92, 0.45, 0.20]
    ball.overlays.update(name=True, velocity=True)
    discrete = scene.create_object("basic.cube", name="Discrete-Time Body", params={"size": 0.25},
                                   position=[-4.5, 2.0, 0.125])
    discrete.linear_velocity = [2.5, 0.0, 6.0]
    discrete.linear_acceleration = [0.0, 0.0, -9.80665]
    discrete.angular_acceleration = [0.0, 0.0, 2.0]
    discrete.color = [0.55, 0.80, 0.35]
    discrete.overlays.update(name=True)
    editor.add_objects([ground, anchor, bob, ball, discrete], text="Formula demo", select=False)
    editor.add_custom_property([ball.id], CustomProperty.create(key="kinetic_energy", name="Kinetic energy",
                                                                vtype="float", quantity="energy").to_dict())
    editor.set_global_parameters([
        {"name": "air_density", "type": "float", "quantity": "density", "value": 1.225,
         "description": "Air density used by Wind Drag"},
        {"name": "wind", "type": "vec3", "quantity": "velocity", "value": [3.0, 1.0, 0.0],
         "description": "Uniform wind velocity"}])
    editor.assign_process("lib.lissajous", TargetFilter(ids=[anchor.id]),
                          {"center": [0.0, 0.0, 2.6], "ax": 1.2, "ay": 0.6, "wx": 0.8, "wy": 1.6})
    editor.assign_process("dyn.gravity", TargetFilter(ids=[bob.id, ball.id]), label="Earth gravity")
    editor.assign_process("lib.tether", TargetFilter(ids=[anchor.id, bob.id]),
                          {"length": 1.2, "stiffness": 400.0, "damping": 1.5})
    editor.assign_process("lib.wind_drag", TargetFilter(ids=[bob.id, ball.id]), {"cd": 0.47, "area": 0.045})
    editor.assign_process("lib.kinetic_energy_monitor", TargetFilter(ids=[ball.id]))
    for pid in ("lib.euler_velocity_update", "lib.euler_position_update", "lib.euler_angular_velocity_update",
                "lib.euler_orientation_update"):
        editor.assign_process(pid, TargetFilter(ids=[discrete.id]))


def _frames_demo(scene: Scene, editor: SceneEditor) -> None:
    """Named reference frames: a turntable carrying a frame, a ball crossing it, a fixed lab frame.

    Open Coordinates ▸ Relative Motion Panel, pick the ball and the 'Deck' frame and run: the ball moves in a
    straight line (zero absolute acceleration) but the deck observer sees it curve under the centripetal and
    Coriolis terms.
    """
    from ..processes.process_base import TargetFilter
    ground = scene.create_object("basic.plane", name="Ground", params={"length": 10, "width": 8})
    ground.color = [0.32, 0.34, 0.38]
    deck = scene.create_object("basic.cylinder", name="Turntable", params={"radius": 2.0, "height": 0.1},
                               position=[0.0, 0.0, 0.05])
    deck.color = [0.33, 0.45, 0.62]
    rider = scene.create_object("basic.cube", name="Rider", params={"size": 0.3}, position=[1.4, 0.0, 0.25])
    rider.color = [0.55, 0.80, 0.35]
    ball = scene.create_object("basic.sphere", name="Ball", params={"radius": 0.15}, position=[-3.5, -0.8, 0.25])
    ball.linear_velocity = [1.0, 0.0, 0.0]
    ball.color = [0.95, 0.50, 0.20]
    ball.overlays.update(name=True, velocity=True)
    editor.add_objects([ground, deck, rider, ball], text="Reference frames demo", select=False)
    editor.reparent([rider.id], deck.id)
    editor.add_frame("Deck", deck.id, [0.0, 0.0, 0.05], axis_length=0.8,
                     description="Rotates with the turntable")
    editor.add_frame("Deck (non-rotating)", deck.id, [0.0, 0.0, 0.05], follow_rotation=False, visible=False,
                     description="Origin on the turntable axis, global orientation")
    editor.add_frame("Lab", None, [-4.0, 3.0, 0.0], axis_length=0.8, color=[0.55, 0.80, 1.0])
    editor.assign_process("rot.fixed_axis", TargetFilter(ids=[deck.id]), {"axis": [0, 0, 1], "angular_speed": 0.8})
    editor.assign_process("kin.free_motion", TargetFilter(ids=[ball.id]))
    editor.scene.set_selection([ball.id])


def _recording_demo(scene: Scene, editor: SceneEditor) -> None:
    """Data recording: a projectile with air drag and a two-mass oscillator, with trajectories and arrows."""
    from ..processes.process_base import TargetFilter
    ground = scene.create_object("basic.plane", name="Ground", params={"length": 14, "width": 8})
    ground.color = [0.32, 0.34, 0.38]
    ball = scene.create_object("basic.sphere", name="Projectile", params={"radius": 0.12}, position=[-6.0, -2.0, 0.12])
    ball.mass_mode, ball.mass_input = "mass", 0.4
    ball.linear_velocity = [7.0, 0.0, 8.0]
    ball.color = [0.95, 0.50, 0.20]
    ball.overlays.update(name=True, velocity=True, velocity_arrow=True)
    left = scene.create_object("body.point_mass", name="Mass A", position=[-1.0, 2.0, 1.0])
    right = scene.create_object("body.point_mass", name="Mass B", position=[1.6, 2.0, 1.0])
    left.mass_input, right.mass_input = 1.0, 2.0
    left.linear_velocity = [1.0, 0.0, 0.0]     # zero total momentum: the pair oscillates without drifting
    right.linear_velocity = [-0.5, 0.0, 0.0]
    editor.add_objects([ground, ball, left, right], text="Recording demo", select=False)
    for path, value in (("trajectory.enabled", True), ("trajectory.arrows", True), ("trajectory.arrow_spacing", 0.25),
                        ("trajectory.time_labels", True), ("trajectory.label_spacing", 0.5)):
        editor.set_property([ball.id], path, value)
    editor.set_property([left.id], "trajectory.enabled", True)
    editor.set_property([left.id], "trajectory.color", [0.35, 0.75, 1.0])
    editor.set_property([right.id], "trajectory.enabled", True)
    editor.set_property([right.id], "trajectory.color", [0.55, 0.9, 0.45])
    editor.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]), label="Earth gravity")
    editor.assign_process("lib.quadratic_drag", TargetFilter(ids=[ball.id]), {"rho": 1.225, "cd": 0.47, "area": 0.045})
    editor.assign_process("dyn.spring", TargetFilter(ids=[left.id, right.id]),
                          {"stiffness": 30.0, "rest_length": 2.0, "damping": 0.2})


PROJECT_TEMPLATES = {
    "empty": ("Empty Scene", None),
    "primitives": ("Primitive Gallery", _primitive_gallery),
    "mechanical": ("Mechanical Demo", _mechanical_demo),
    "simulation": ("Simulation Demo", _simulation_demo),
    "formulas": ("Formula Processes Demo", _formula_demo),
    "frames": ("Reference Frames Demo", _frames_demo),
    "recording": ("Data Recording & Plots Demo", _recording_demo),
}
