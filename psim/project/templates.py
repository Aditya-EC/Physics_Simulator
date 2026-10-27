"""Built-in project templates (File ▸ New from Template)."""
from __future__ import annotations

import math

from .. import numerics
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


def _physics_demo(scene: Scene, editor: SceneEditor) -> None:
    """Rigid bodies: a bouncing ball, a box stack, a hinged pendulum with a limit, a rope chain and a slider."""
    from ..processes.process_base import TargetFilter
    ground = scene.create_object("basic.plane", name="Ground", params={"length": 16, "width": 10})
    ground.color = [0.30, 0.32, 0.36]
    ball = scene.create_object("basic.sphere", name="Bouncing Ball", params={"radius": 0.2}, position=[-5.0, 0.0, 3.0])
    ball.color = [0.95, 0.45, 0.20]
    ball.linear_velocity = [1.0, 0.0, 0.0]
    stack = [scene.create_object("body.rigid_box", name=f"Crate {k + 1}", position=[-2.0, 0.0, 0.25 + 0.5 * k],
                                 params={"length": 0.5, "width": 0.5, "height": 0.5}) for k in range(3)]
    for k, box in enumerate(stack):
        box.color = [0.80, 0.62 - 0.1 * k, 0.35]
    striker = scene.create_object("basic.sphere", name="Striker", params={"radius": 0.15}, position=[-4.0, 0.0, 0.15])
    striker.linear_velocity = [4.0, 0.0, 0.0]
    striker.color = [0.35, 0.75, 1.0]
    post = scene.create_object("basic.cylinder", name="Post", params={"radius": 0.08, "height": 3.0},
                               position=[1.5, 0.0, 1.5])
    post.color = [0.55, 0.57, 0.62]
    arm = scene.create_object("body.rigid_box", name="Pendulum Arm", position=[2.3, 0.0, 3.0],
                              params={"length": 1.6, "width": 0.12, "height": 0.12})
    arm.color = [0.55, 0.85, 0.45]
    links = [scene.create_object("basic.sphere", name=f"Chain Link {k + 1}", params={"radius": 0.12},
                                 position=[4.5 + 0.5 * k, -2.0, 3.0]) for k in range(3)]
    for link in links:
        link.color = [0.85, 0.75, 0.95]
    rail_cart = scene.create_object("body.rigid_box", name="Slider", position=[-1.0, 3.0, 0.6],
                                    params={"length": 0.4, "width": 0.3, "height": 0.2})
    rail_cart.color = [0.95, 0.80, 0.30]
    editor.add_objects([ground, ball, *stack, striker, post, arm, *links, rail_cart], text="Physics demo",
                       select=False)
    colliding = [ground, ball, striker, *links]
    for o in colliding:
        editor.set_property([o.id], "physics.collider", "auto")
    editor.set_property([ball.id], "physics.restitution", 0.75)
    editor.set_property([arm.id], "physics.show_collider", True)
    editor.set_property([arm.id], "trajectory.enabled", True)
    editor.set_property([links[-1].id], "trajectory.enabled", True)
    # hinge at the top of the post, swinging freely about Y (the slider below shows limits and a motor)
    editor.add_joint("revolute", arm.id, None, [1.5, 0.0, 3.0], [0.0, 1.0, 0.0], name="Hinge")
    # rope chain hanging from a fixed point
    prev, prev_point = None, [4.0, -2.0, 3.0]
    for k, link in enumerate(links):
        p = [4.5 + 0.5 * k, -2.0, 3.0]
        editor.add_joint("distance", link.id, prev, p, name=f"Rope {k + 1}", anchor_world_b=prev_point, rope=True)
        prev, prev_point = link.id, p
    # slider on a horizontal rail, driven by a motor against its travel limits
    editor.add_joint("prismatic", rail_cart.id, None, [-1.0, 3.0, 0.6], [1.0, 0.0, 0.0], name="Rail",
                     limit_enabled=True, lower=-1.5, upper=1.5, motor_enabled=True, motor_speed=0.8, motor_max=40.0)
    dynamic = [ball, *stack, striker, arm, *links, rail_cart]
    editor.assign_process("dyn.gravity", TargetFilter(ids=[o.id for o in dynamic]), label="Earth gravity")


def _measurements_demo(scene: Scene, editor: SceneEditor) -> None:
    """A pendulum on a rod with live angle and distance measurements (a stop condition is in the example file)."""
    from ..processes.process_base import TargetFilter
    ground = scene.create_object("basic.plane", name="Ground", params={"length": 8, "width": 6})
    ground.color = [0.30, 0.32, 0.36]
    pivot = scene.create_object("basic.sphere", name="Pivot", params={"radius": 0.06}, position=[0.0, 0.0, 3.0])
    pivot.color = [0.85, 0.85, 0.9]
    plumb = scene.create_object("frame.marker", name="Plumb Point", position=[0.0, 0.0, 1.5])
    angle0 = math.radians(60)
    bob = scene.create_object("basic.sphere", name="Bob", params={"radius": 0.15},
                              position=[-1.5 * math.sin(angle0), 0.0, 3.0 - 1.5 * math.cos(angle0)])
    bob.color = [0.95, 0.55, 0.20]
    target = scene.create_object("basic.cube", name="Target", params={"size": 0.3}, position=[2.0, 0.0, 0.15])
    target.color = [0.35, 0.75, 1.0]
    editor.add_objects([ground, pivot, plumb, bob, target], text="Measurements demo", select=False)
    editor.set_property([bob.id], "trajectory.enabled", True)
    editor.add_joint("distance", bob.id, None, list(scene.world_pose(bob.id)[0]), name="Rod",
                     anchor_world_b=[0.0, 0.0, 3.0])
    editor.assign_process("dyn.gravity", TargetFilter(ids=[bob.id]), label="Earth gravity")
    editor.add_measurement("angle", [plumb.id, pivot.id, bob.id], name="Swing angle")
    editor.add_measurement("distance", [bob.id, target.id], name="Bob to target")


def _mechanisms_demo(scene: Scene, editor: SceneEditor) -> None:
    """Springs, breakable joints, convex parts and contact materials, one station each along X."""
    from ..processes.process_base import TargetFilter
    ground = scene.create_object("basic.plane", name="Ground", params={"length": 22, "width": 10})
    ground.color = [0.30, 0.32, 0.36]
    # 1. a ball falls through the hole of a ring (the torus collides as convex parts, not as its filled hull)
    ring = scene.create_object("basic.torus", name="Ring", position=[-8.0, 0.0, 0.15])
    ring.color = [0.65, 0.55, 0.85]
    marble = scene.create_object("basic.sphere", name="Marble", params={"radius": 0.12}, position=[-8.0, 0.0, 2.0])
    marble.color = [0.95, 0.45, 0.20]
    # 2. a platform on a vertical slider with a spring and damper catches a ball
    platform = scene.create_object("basic.cuboid", name="Sprung Platform", position=[-4.5, 0.0, 1.0],
                                   params={"length": 1.0, "width": 1.0, "height": 0.1})
    platform.color = [0.35, 0.75, 1.0]
    dropped = scene.create_object("basic.sphere", name="Dropped Ball", params={"radius": 0.15},
                                  position=[-4.5, 0.0, 3.0])
    dropped.color = [0.95, 0.80, 0.30]
    # 3. a door on a vertical hinge with a torsion spring, pushed open
    door = scene.create_object("basic.cuboid", name="Spring Door", position=[-1.1, 0.0, 0.7],
                               params={"length": 0.8, "width": 0.05, "height": 1.2})
    door.color = [0.55, 0.85, 0.45]
    door.angular_velocity = [0.0, 0.0, 2.0]
    # 4. a crate on a breakable rope; a heavy ball falls on it and the rope snaps
    crate = scene.create_object("basic.cube", name="Hanging Crate", params={"size": 0.4}, position=[1.5, 0.0, 1.8])
    crate.color = [0.80, 0.62, 0.35]
    hammer = scene.create_object("basic.sphere", name="Hammer", params={"radius": 0.15}, position=[1.55, 0.0, 4.0])
    hammer.color = [0.75, 0.75, 0.80]
    # 5. two blocks on a 25° slope: the one with high static friction stays, the other slides
    tilt = math.radians(25)
    ramp = scene.create_object("basic.cuboid", name="Ramp", position=[5.0, 0.0, 1.0],
                               params={"length": 3.0, "width": 1.6, "height": 0.1})
    ramp.color = [0.55, 0.57, 0.62]
    n = [math.sin(tilt), 0.0, math.cos(tilt)]
    blocks = []
    for k, (name, y) in enumerate((("Sticky Block", -0.4), ("Slippery Block", 0.4))):
        b = scene.create_object("basic.cube", name=name, params={"size": 0.3},
                                position=[5.0 + 0.2 * n[0], y, 1.0 + 0.2 * n[2]])
        b.color = [0.95, 0.55, 0.20] if k == 0 else [0.35, 0.75, 1.0]
        blocks.append(b)
    # 6. two balls rolling side by side; one has rolling resistance
    rollers = []
    for k, (name, x) in enumerate((("Rolling Ball (b = 5 mm)", 7.4), ("Rolling Ball (b = 0)", 8.2))):
        r = 0.15
        b = scene.create_object("basic.sphere", name=name, params={"radius": r}, position=[x, -3.5, r])
        b.linear_velocity = [0.0, 2.5, 0.0]
        b.angular_velocity = [-2.5 / r, 0.0, 0.0]          # rolling without slipping
        b.color = [0.85, 0.75, 0.95] if k == 0 else [0.95, 0.95, 0.95]
        rollers.append(b)
    objs = [ground, ring, marble, platform, dropped, door, crate, hammer, ramp, *blocks, *rollers]
    editor.add_objects(objs, text="Mechanisms demo", select=False)
    for o in objs:
        editor.set_property([o.id], "physics.collider", "auto")
    editor.set_property([ramp.id], "orientation", list(numerics.quat_from_axis_angle([0, 1, 0], tilt)))
    for b in blocks:
        editor.set_property([b.id], "orientation", list(numerics.quat_from_axis_angle([0, 1, 0], tilt)))
    for o in (platform, door, crate):
        editor.set_property([o.id], "density", 500.0)       # wood
    editor.set_property([ring.id], "physics.show_collider", True)
    editor.set_property([dropped.id], "physics.restitution", 0.6)
    editor.set_property([ramp.id], "physics.friction", 0.3)
    editor.set_property([ramp.id], "physics.static_friction", 0.6)
    for b, mu_s in zip(blocks, (0.6, 0.3)):
        editor.set_property([b.id], "physics.friction", 0.3)
        editor.set_property([b.id], "physics.static_friction", mu_s)
    editor.set_property([rollers[0].id], "physics.rolling_resistance", 0.005)
    for o in (marble, dropped, *rollers):
        editor.set_property([o.id], "trajectory.enabled", True)
    editor.add_joint("prismatic", platform.id, None, [-4.5, 0.0, 1.0], [0.0, 0.0, 1.0], name="Suspension",
                     spring_stiffness=6000.0, spring_damping=150.0)
    editor.add_joint("revolute", door.id, None, [-1.5, 0.0, 0.7], [0.0, 0.0, 1.0], name="Door Hinge",
                     spring_stiffness=40.0, spring_damping=1.5)
    crate_mass = crate.mass_properties()["mass"]
    editor.add_joint("distance", crate.id, None, [1.5, 0.0, 2.0], name="Breakable Rope",
                     anchor_world_b=[1.5, 0.0, 3.0], rope=True, break_force=3.0 * crate_mass * 9.80665)
    dynamic = [marble, platform, dropped, door, crate, hammer, *blocks, *rollers]
    editor.assign_process("dyn.gravity", TargetFilter(ids=[o.id for o in dynamic]), label="Earth gravity")


PROJECT_TEMPLATES = {
    "empty": ("Empty Scene", None),
    "primitives": ("Primitive Gallery", _primitive_gallery),
    "mechanical": ("Mechanical Demo", _mechanical_demo),
    "simulation": ("Simulation Demo", _simulation_demo),
    "formulas": ("Formula Processes Demo", _formula_demo),
    "frames": ("Reference Frames Demo", _frames_demo),
    "recording": ("Data Recording & Plots Demo", _recording_demo),
    "physics": ("Rigid Bodies, Collisions & Joints Demo", _physics_demo),
    "measurements": ("Measurements & Stop Conditions Demo", _measurements_demo),
    "mechanisms": ("Springs, Breakable Joints & Contact Materials Demo", _mechanisms_demo),
}
