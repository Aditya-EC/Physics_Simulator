"""Reference frames: kinematics against finite differences, named frames in the scene, undo and project files."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from psim import numerics
from psim.coordinates.frames import FrameError, validate_frames
from psim.coordinates.kinematics import (FrameState, RigidState, express, point_from_frame, point_in_frame,
                                         relative_motion, relative_rotation, rotation_of, unexpress)
from psim.coordinates.scene_kinematics import reference_state, rigid_state


def rz(a):
    return numerics.quat_from_axis_angle([0, 0, 1], a)


def rx(a):
    return numerics.quat_from_axis_angle([1, 0, 0], a)


A, B = 0.9, 0.35          # body rotation R(t) = Rz(A t) · Rx(B t²): ω and α are not parallel
R0 = np.array([0.4, -0.2, 0.3])   # body origin relative to the COM (body axes)
OFF = np.array([0.5, 0.25, -0.1])  # frame origin on the body (body axes)
QOFF = numerics.quat_from_axis_angle(np.array([1.0, 2.0, 0.5]) / np.linalg.norm([1.0, 2.0, 0.5]), 0.6)


def body(t: float) -> RigidState:
    q = np.asarray(numerics.quat_multiply(rz(A * t), rx(B * t * t)), float)
    ez = np.array([0, 0, 1.0])
    ex_rot = np.asarray(numerics.quat_rotate(rz(A * t), [1, 0, 0]), float)
    w = A * ez + 2 * B * t * ex_rot
    alpha = 2 * B * ex_rot + 2 * B * t * np.cross(A * ez, ex_rot)
    com = np.array([math.cos(t), 0.5 * t * t, math.sin(2 * t)])
    v = np.array([-math.sin(t), t, 2 * math.cos(2 * t)])
    a = np.array([-math.cos(t), 1.0, -4 * math.sin(2 * t)])
    origin = com + np.asarray(numerics.quat_rotate(q, R0), float)
    return RigidState(origin, q, com, v, w, a, alpha)


def point(t: float):
    p = np.array([2 * math.sin(t), t ** 3 / 3, math.cos(3 * t)])
    v = np.array([2 * math.cos(t), t * t, -3 * math.sin(3 * t)])
    a = np.array([-2 * math.sin(t), 2 * t, -9 * math.cos(3 * t)])
    return p, v, a


@pytest.mark.parametrize("follow", [True, False])
def test_relative_motion_matches_finite_differences(follow):
    t, h = 0.8, 1e-4

    def frame(tt):
        return body(tt).attached(OFF, QOFF, follow)

    def coords(tt):
        return point_in_frame(point(tt)[0], frame(tt))

    rm = relative_motion(*point(t), frame(t))
    np.testing.assert_allclose(rm.position, coords(t), atol=1e-12)
    vel_fd = (coords(t + h) - coords(t - h)) / (2 * h)
    acc_fd = (coords(t + h) - 2 * coords(t) + coords(t - h)) / h ** 2
    np.testing.assert_allclose(rm.velocity_relative, vel_fd, atol=1e-7)
    np.testing.assert_allclose(rm.acceleration_relative, acc_fd, atol=2e-5)
    # the attached frame's own origin motion must also match finite differences
    o = lambda tt: frame(tt).origin  # noqa: E731
    np.testing.assert_allclose(frame(t).velocity, (o(t + h) - o(t - h)) / (2 * h), atol=1e-7)
    np.testing.assert_allclose(frame(t).acceleration, (o(t + h) - 2 * o(t) + o(t - h)) / h ** 2, atol=2e-5)
    # representation is only a rotation of the absolute vectors
    R = frame(t).rotation
    np.testing.assert_allclose(rm.velocity_components, R.T @ point(t)[1], atol=1e-12)
    np.testing.assert_allclose(rm.acceleration_components, R.T @ point(t)[2], atol=1e-12)
    # the breakdown adds up: a = a_rel + a_F + Euler + centripetal + Coriolis (all in frame components)
    total = (rm.acceleration_relative + rm.origin_acceleration + rm.euler_acceleration + rm.centripetal_acceleration
             + rm.coriolis_acceleration)
    np.testing.assert_allclose(total, rm.acceleration_components, atol=1e-12)
    if not follow:
        assert np.allclose(rm.coriolis_acceleration, 0) and np.allclose(rm.euler_acceleration, 0)


def test_relative_rotation_matches_finite_differences():
    t, h = 0.6, 1e-4
    frame = lambda tt: body(tt).attached(OFF, QOFF, True)  # noqa: E731

    def other(tt):  # a second body spinning about a tilted axis with quadratic angle
        axis = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
        return numerics.quat_from_axis_angle(axis, 1.3 * tt + 0.4 * tt * tt), axis * (1.3 + 0.8 * tt), axis * 0.8

    def r_rel(tt):
        return frame(tt).rotation.T @ rotation_of(other(tt)[0])

    rr = relative_rotation(*other(t), frame(t))
    np.testing.assert_allclose(rotation_of(rr.orientation), r_rel(t), atol=1e-12)
    skew = (r_rel(t + h) - r_rel(t - h)) / (2 * h) @ r_rel(t).T
    w_fd = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    np.testing.assert_allclose(rr.angular_velocity_relative, w_fd, atol=1e-7)

    def w_rel(tt):
        return relative_rotation(*other(tt), frame(tt)).angular_velocity_relative
    np.testing.assert_allclose(rr.angular_acceleration_relative, (w_rel(t + h) - w_rel(t - h)) / (2 * h), atol=1e-7)


def test_point_fixed_in_rotating_frame_is_at_rest_relative_to_it():
    t = 1.1
    rb = body(t)
    f = rb.attached(OFF, QOFF, True)
    p_local = np.array([0.3, -0.7, 0.2])
    p = point_from_frame(p_local, f)
    rm = relative_motion(p, rb.point_velocity(p), rb.point_acceleration(p), f)
    np.testing.assert_allclose(rm.position, p_local, atol=1e-12)
    np.testing.assert_allclose(rm.velocity_relative, 0, atol=1e-12)
    np.testing.assert_allclose(rm.acceleration_relative, 0, atol=1e-12)


def test_representation_versus_relative_velocity_on_a_turntable():
    """A point at rest at (1,0,0) seen from a frame spinning at 2 rad/s about z, at the instant the axes coincide."""
    f = FrameState(np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3), np.array([0, 0, 2.0]), np.zeros(3), np.zeros(3))
    rm = relative_motion([1, 0, 0], [0, 0, 0], [0, 0, 0], f)
    np.testing.assert_allclose(rm.velocity_components, [0, 0, 0])     # the point is not moving
    np.testing.assert_allclose(rm.velocity_relative, [0, -2, 0])      # but the observer sees it sweep backwards
    np.testing.assert_allclose(rm.centripetal_acceleration, [-4, 0, 0])
    np.testing.assert_allclose(rm.coriolis_acceleration, [0, 0, 0] + 2 * np.cross([0, 0, 2], [0, -2, 0]))
    np.testing.assert_allclose(rm.acceleration_relative, [-4, 0, 0])  # observed circular motion: -ω²r
    v = np.array([0.3, -1.2, 2.0])
    np.testing.assert_allclose(unexpress(express(v, body(0.4).frame()), body(0.4).frame()), v, atol=1e-12)


def test_unknown_accelerations_are_reported_as_unknown():
    f = RigidState(np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3), np.array([0, 0, 1.0])).frame()
    rm = relative_motion([1, 0, 0], [0, 1, 0], [0, 0, 0], f)
    assert rm.acceleration_relative is None and rm.coriolis_acceleration is None
    np.testing.assert_allclose(rm.velocity_relative, [0, 0, 0])


# ---- named frames in the scene -------------------------------------------------------------------------------

def test_frame_validation():
    good = {"id": "FRM-0001", "name": "Lab", "position": [1, 2, 3], "orientation": [2, 0, 0, 0]}
    f = validate_frames([good])[0]
    assert f.orientation == [1.0, 0.0, 0.0, 0.0] and f.parent is None
    for bad, msg in (({**good, "id": "F1"}, "Invalid"), ({**good, "name": " "}, "needs a name"),
                     ({**good, "position": [1, 2]}, "position"), ({**good, "orientation": [0, 0, 0, 0]}, "non-zero"),
                     ({**good, "script": "x"}, "unknown field"), ({**good, "axis_length": -1}, "positive")):
        with pytest.raises(FrameError, match=msg):
            validate_frames([bad])
    with pytest.raises(FrameError, match="Duplicate reference frame name"):
        validate_frames([good, {**good, "id": "FRM-0002", "name": "lab"}])


def test_attached_frames_follow_their_parent(scene):
    s, e = scene
    asm = e.add_new("group.assembly", position=[1, 0, 0])
    asm.set_orientation(rz(math.pi / 2))
    rot = e.add_new("basic.cube", position=[0, 0, 0])
    fixed = e.add_frame("Lab", None, [0, 0, 2], rz(math.pi / 4))
    att = e.add_frame("Arm", asm.id, [1, 0, 0])
    trans = e.add_frame("Carried", asm.id, [1, 0, 0], follow_rotation=False)
    assert [f.id for f in s.frames] == [fixed, att, trans]
    p, q = s.frame_world_pose(att)
    np.testing.assert_allclose(p, [1, 1, 0], atol=1e-12)
    np.testing.assert_allclose(rotation_of(q), rotation_of(rz(math.pi / 2)), atol=1e-12)
    np.testing.assert_allclose(rotation_of(s.frame_world_pose(trans)[1]), np.eye(3), atol=1e-12)
    # poses of objects can be read and edited in a named frame
    e.set_pose_in_frame(rot.id, att, np.array([0.0, 2.0, 0.0]), None)
    np.testing.assert_allclose(s.world_pose(rot.id)[0], [-1, 1, 0], atol=1e-12)
    np.testing.assert_allclose(s.pose_in_frame(rot.id, att)[0], [0, 2, 0], atol=1e-12)
    np.testing.assert_allclose(s.pose_in_frame(rot.id, fixed)[0],
                               rotation_of(rz(math.pi / 4)).T @ np.array([-1, 1, -2]), atol=1e-12)
    # moving the parent moves the frame
    e.set_property([asm.id], "position", [5, 0, 0])
    np.testing.assert_allclose(s.frame_world_pose(att)[0], [5, 1, 0], atol=1e-12)


def test_frame_edits_are_undoable_and_ids_not_reused(scene):
    s, e = scene
    o = e.add_new("basic.sphere")
    f1 = e.add_frame("A", o.id)
    f2 = e.add_frame("B")
    e.update_frame(f1, name="A2", position=[0, 0, 1])
    assert s.frame(f1).name == "A2"
    e.remove_frames([f2])
    assert s.frame(f2) is None
    e.stack.undo()
    assert s.frame(f2) is not None
    e.stack.undo()
    assert s.frame(f1).name == "A" and s.frame(f1).position == [0, 0, 0]
    e.stack.redo()
    e.stack.redo()
    assert s.frame(f2) is None
    assert e.add_frame("C") == "FRM-0003"
    with pytest.raises(Exception, match="missing object"):
        e.add_frame("D", "OBJ-999999")
    with pytest.raises(FrameError, match="Duplicate"):
        e.add_frame("c")
    # deleting the parent leaves the frame detached (reported), and undo restores it
    e.delete([o.id])
    assert any("missing object" in p for p in s.validate())
    with pytest.raises(Exception, match="missing object"):
        s.frame_world_pose(f1)
    e.stack.undo()
    assert s.validate() == []


def test_frames_saved_in_project(scene):
    from psim.project import serialization as ser
    from psim.scene.scene_manager import Scene
    from psim.units.unit_registry import UnitRegistry
    s, e = scene
    o = e.add_new("basic.cube")
    e.add_frame("Sensor", o.id, [0, 0, 0.5], rx(0.3), follow_rotation=False, axis_length=0.3)
    d = json.loads(json.dumps(ser.project_to_dict(s, UnitRegistry(), ser.ProjectData())))
    s2 = Scene()
    data = ser.load_project_dict(d, s2, UnitRegistry())
    assert s2.frames_snapshot() == s.frames_snapshot() and data.warnings == []
    assert s2.allocate_frame_id() == "FRM-0002"
    d["objects"] = [x for x in d["objects"] if x["id"] != o.id]
    data = ser.load_project_dict(d, Scene(), UnitRegistry())
    assert any("attached to missing object" in w for w in data.warnings)
    d["reference_frames"][0]["id"] = "bad"
    with pytest.raises(ser.ProjectError):
        ser.load_project_dict(d, Scene(), UnitRegistry())


def test_scene_kinematics_follow_simulated_ancestors(scene):
    s, e = scene
    asm = e.add_new("group.assembly", position=[0, 0, 1])
    child = e.add_new("body.point_mass", position=[2, 0, 1])
    e.reparent([child.id], asm.id)
    lone = e.add_new("body.point_mass", position=[0, 3, 0])
    lone.linear_velocity = [9, 9, 9]
    asm.linear_velocity = [1, 0, 0]
    asm.angular_velocity = [0, 0, 0.5]
    fid = e.add_frame("Tip", asm.id, [2, 0, 0])
    # before a run: velocities are the initial conditions, accelerations unknown
    assert rigid_state(s, lone.id).acceleration is None
    np.testing.assert_allclose(rigid_state(s, lone.id).velocity, [9, 9, 9])
    # during a run: the assembly is simulated, the child moves with it, the lone object is at rest
    s.simulating = True
    s.runtime = {asm.id: {"acceleration": np.array([0, 0, -1.0]), "angular_acceleration": np.zeros(3)}}
    np.testing.assert_allclose(rigid_state(s, child.id).velocity, [1, 1, 0])
    np.testing.assert_allclose(rigid_state(s, child.id).acceleration, [-0.5, 0, -1])
    np.testing.assert_allclose(rigid_state(s, lone.id).velocity, 0)
    tip = reference_state(s, fid)
    np.testing.assert_allclose(tip.velocity, [1, 1, 0])
    rm = relative_motion(*[rigid_state(s, child.id).com, rigid_state(s, child.id).velocity,
                           rigid_state(s, child.id).acceleration], tip)
    np.testing.assert_allclose(rm.velocity_relative, 0, atol=1e-12)    # the child is carried with the tip frame
    np.testing.assert_allclose(rm.acceleration_relative, 0, atol=1e-12)


# ---- frames in formulas and in the engine ---------------------------------------------------------------------

from psim.processes.formula_process import FormulaProcessSpec, build_process_def, compile_spec, run_tests  # noqa: E402
from psim.processes.process_base import TargetFilter  # noqa: E402
from psim.processes.process_registry import process_registry  # noqa: E402

_injected: list[str] = []


@pytest.fixture(autouse=True)
def _remove_injected():
    yield
    reg = process_registry()
    while _injected:
        pid = _injected.pop()
        reg._defs.pop(pid, None)
        reg.sources.pop(pid, None)


def register_temp(spec_dict) -> str:
    reg = process_registry()
    spec = FormulaProcessSpec.from_dict(spec_dict)
    cp, issues = compile_spec(spec)
    assert cp is not None, [str(i) for i in issues]
    reg._defs[spec.process_id] = build_process_def(cp)  # test-only injection into the sealed registry
    reg.sources[spec.process_id] = "test"
    _injected.append(spec.process_id)
    return spec.process_id


def check_issue(script, kind="contribution", needle=""):
    cp, issues = compile_spec(FormulaProcessSpec.from_dict({"process_id": "t.x", "name": "x", "kind": kind,
                                                            "requires_mass": True, "script": script}))
    assert cp is None and needle in issues[0].message, [str(i) for i in issues]


def test_frame_type_checking():
    check_issue('force = frame("FRM-0001").mass * 1[m/s^2]', needle="Reference frames have no attribute 'mass'")
    check_issue("force = rel_velocity(self.position, self.position, self) * 1[kg/s]", needle="v must be a velocity")
    check_issue("force = frame(1)", needle="string literal")
    check_issue('force = rel_position(frame("FRM-0001"), self) * 1[N/m]', needle="expects an object")
    check_issue('force = rel_velocity(self, frame("FRM-0001"))', needle="'force' needs a 3D vector [L·M·T^-2]")


TURNTABLE = {
    "process_id": "t.observed", "name": "Observed velocity", "kind": "update",
    "custom": [{"key": "v_rel", "type": "vec3", "quantity": "velocity"},
               {"key": "v_comp", "type": "vec3", "quantity": "velocity"}],
    "script": 'let F = frame("FRM-0001")\n'
              "custom.v_rel = rel_velocity(self, F)\n"
              "custom.v_comp = to_local(F, self.velocity)\n",
    "tests": [{"name": "turntable", "objects": {"self": {"position": [1, 0, 0]}},
               "frames": {"FRM-0001": {"angular_velocity": [0, 0, 2]}},
               "expect": {"custom.v_rel": [0, -2, 0], "custom.v_comp": [0, 0, 0]}},
              {"name": "missing frame", "objects": {"self": {}}, "expect_error": "Missing reference frame"}],
}


def test_frame_functions_in_process_test_cases():
    cp, issues = compile_spec(FormulaProcessSpec.from_dict(TURNTABLE))
    assert cp is not None, [str(i) for i in issues]
    assert cp.frame_ids == {"FRM-0001"}
    results = run_tests(cp)
    assert all(r.passed for r in results), [(r.name, r.message) for r in results]


def _engine(scene, **settings):
    from psim.processes.process_scheduler import build_plan
    from psim.simulation.settings import SimulationSettings
    from psim.simulation.simulation_engine import SimulationEngine
    from psim.simulation.state import capture_bodies
    from psim.simulation.world import EngineWorld, capture_frames, capture_static, global_values
    s, _ = scene
    plan = build_plan(s, s.assignments)
    assert plan.ok, [d.message for d in plan.errors]
    bodies, state = capture_bodies(s, plan.body_ids)
    statics = capture_static(s)
    world = EngineWorld(statics, plan.body_ids, global_values(s), settings.get("delta_time", 0.01),
                        capture_frames(s, statics))
    return SimulationEngine(SimulationSettings(**settings), plan, bodies, state, world)


def test_formula_sees_frames_and_carried_children_during_a_run(scene):
    """A rotating assembly carries a child and a named frame; a monitor reads them through formulas."""
    from psim.processes.process_scheduler import build_plan
    from psim.properties.custom_properties import CustomProperty
    s, e = scene
    asm = e.add_new("group.assembly")
    child = e.add_new("body.point_mass", position=[1, 0, 0])
    e.reparent([child.id], asm.id)
    fid = e.add_frame("Rim", asm.id, [1, 0, 0])
    probe = e.add_new("body.point_mass", position=[0, 5, 0])
    for key in ("v_rel", "v_child", "frame_v"):
        probe.add_custom(CustomProperty.create(key=key, name=key, vtype="vec3", quantity="velocity"))
    e.assign_process("rot.fixed_axis", TargetFilter(ids=[asm.id]), {"axis": [0, 0, 1], "angular_speed": 2.0})
    pid = register_temp({
        "process_id": "t.monitor", "name": "Monitor", "kind": "update",
        "custom": [{"key": k, "type": "vec3", "quantity": "velocity"} for k in ("v_rel", "v_child", "frame_v")],
        "script": f'let F = frame("{fid}")\n'
                  f'custom.v_child = object("{child.id}").velocity\n'
                  f'custom.frame_v = F.velocity\n'
                  f'custom.v_rel = rel_velocity(object("{child.id}"), F)\n'})
    e.assign_process(pid, TargetFilter(ids=[probe.id]))
    eng = _engine(scene, delta_time=0.01, duration=1.0, integrator="rk4")
    eng.step()
    snap = eng.snapshot()
    out = snap.custom[probe.id]
    # the custom values are computed from the state at the start of the step (t = 0)
    np.testing.assert_allclose(out["v_child"], [0, 2, 0], atol=1e-9)   # ω × r of the carried child
    np.testing.assert_allclose(out["frame_v"], [0, 2, 0], atol=1e-9)
    np.testing.assert_allclose(out["v_rel"], [0, 0, 0], atol=1e-9)    # the child is fixed in the rim frame
    # a frame that does not exist is reported before the run
    e.remove_frames([fid])
    assert any("missing reference frame" in d.message for d in build_plan(s, s.assignments).errors)


def test_unsimulated_objects_are_at_rest_in_formulas(scene):
    from psim.properties.custom_properties import CustomProperty
    s, e = scene
    still = e.add_new("body.point_mass", position=[3, 0, 0])
    still.linear_velocity = [7, 0, 0]          # an initial condition, but no process moves it
    mover = e.add_new("body.point_mass")
    mover.add_custom(CustomProperty.create(key="seen", name="seen", vtype="vec3", quantity="velocity"))
    pid = register_temp({"process_id": "t.watch", "name": "Watch", "kind": "update",
                         "custom": [{"key": "seen", "type": "vec3", "quantity": "velocity"}],
                         "script": f'custom.seen = object("{still.id}").velocity'})
    e.assign_process("kin.free_motion", TargetFilter(ids=[mover.id]))
    e.assign_process(pid, TargetFilter(ids=[mover.id]))
    eng = _engine(scene, delta_time=0.01, duration=0.1)
    eng.step()
    np.testing.assert_allclose(eng.snapshot().custom[mover.id]["seen"], [0, 0, 0])
