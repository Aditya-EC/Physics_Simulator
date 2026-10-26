"""Built-in approved processes (Phase 3 kinematics and dynamics set).

Contribution processes have the signature ``evaluate(ctx, idx, params)`` and
add to ``ctx.acc`` / ``ctx.alpha`` through the StepContext helpers.
Prescribed processes have the signature ``evaluate(t, idx, params, bodies)``
and return a dict with ``position`` and ``velocity`` (origin of the object,
global frame) and/or ``orientation`` and ``angular_velocity``.

State convention: the integrated translational state is the centre of mass
(COM); ``linear_velocity`` of an object is the velocity of its COM. For
objects without mass properties the COM is the object origin.
"""
from __future__ import annotations

import functools

import numpy as np

from .. import numerics
from ..properties.property_system import PropertyDef, ValueType
from .process_base import W_FORCE, W_ORIENTATION, W_POSITION, W_TORQUE, ProcessDef
from .process_registry import ProcessRegistry

V3 = ValueType.VEC3
F = ValueType.FLOAT


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([0.0, 0.0, 1.0])


# ---- contribution processes ------------------------------------------------------------

def _free_motion(ctx, idx, params) -> None:
    """No contribution: marks the targets for integration of their current velocities."""


def _constant_acceleration(ctx, idx, params) -> None:
    ctx.add_acceleration(idx, ctx.bodies.linear_acceleration[idx])
    ctx.add_angular_acceleration(idx, ctx.bodies.angular_acceleration[idx])


def _gravity(ctx, idx, params) -> None:
    ctx.add_acceleration(idx, np.asarray(params["g"], float))


def _applied_force(ctx, idx, params) -> None:
    ctx.add_force(idx, ctx.bodies.force[idx])
    if params["include_torque"]:
        ctx.add_torque(idx, ctx.bodies.torque[idx])


def _damping(ctx, idx, params) -> None:
    ctx.add_force(idx, -params["linear"] * ctx.v[idx])
    if params["angular"] > 0:
        ctx.add_torque(idx, -params["angular"] * ctx.w[idx])


def _spring(ctx, idx, params) -> None:
    """Linear spring-damper between the centres of mass of exactly two bodies."""
    a, b = idx[0], idx[1]
    d = ctx.x[b] - ctx.x[a]
    length = float(np.linalg.norm(d))
    if length < 1e-12:
        return
    n = d / length
    rel_speed = float((ctx.v[b] - ctx.v[a]) @ n)
    f = (params["stiffness"] * (length - params["rest_length"]) + params["damping"] * rel_speed) * n
    ctx.add_force(np.array([a]), f[None, :])
    ctx.add_force(np.array([b]), -f[None, :])


# ---- prescribed processes ------------------------------------------------------------------

@functools.lru_cache(maxsize=256)
def _circle_basis(axis_t: tuple) -> tuple[np.ndarray, np.ndarray]:
    # e1: global X projected into the circle's plane (Y if the axis is along X); e2 = axis × e1,
    # so for axis +Z the motion starts at +X (phase 0) and runs counter-clockwise towards +Y.
    axis = _unit(axis_t)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.99 else np.array([0.0, 1.0, 0.0])
    e1 = ref - axis * float(ref @ axis)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


def _circular(t, idx, params, bodies) -> dict:
    e1, e2 = _circle_basis(tuple(params["axis"]))
    theta = params["phase"] + params["angular_speed"] * t
    r = params["radius"]
    p = np.asarray(params["center"], float) + r * (np.cos(theta) * e1 + np.sin(theta) * e2)
    v = r * params["angular_speed"] * (-np.sin(theta) * e1 + np.cos(theta) * e2)
    n = len(idx)
    return {"position": np.tile(p, (n, 1)), "velocity": np.tile(v, (n, 1))}


def _oscillation(t, idx, params, bodies) -> dict:
    d = _unit(params["direction"])
    om = 2 * np.pi * params["frequency"]
    s = params["amplitude"] * np.sin(om * t + params["phase"])
    ds = params["amplitude"] * om * np.cos(om * t + params["phase"])
    return {"position": bodies.origin0[idx] + s * d, "velocity": np.tile(ds * d, (len(idx), 1))}


def _rotation_fixed_axis(t, idx, params, bodies) -> dict:
    axis = _unit(params["axis"])
    w = axis * params["angular_speed"]
    dq = numerics.quat_from_axis_angle(axis, params["angular_speed"] * t)
    q = np.array([numerics.quat_multiply(dq, q0) for q0 in bodies.q0[idx]])
    return {"orientation": q, "angular_velocity": np.tile(w, (len(idx), 1))}


def _rotation_local_axis(t, idx, params, bodies) -> dict:
    axis = _unit(params["axis"])
    dq = numerics.quat_from_axis_angle(axis, params["angular_speed"] * t)
    q = np.array([numerics.quat_multiply(q0, dq) for q0 in bodies.q0[idx]])  # body-fixed axis
    w = numerics.quat_rotate_batch(q, np.tile(axis * params["angular_speed"], (len(idx), 1)))
    return {"orientation": q, "angular_velocity": w}


def register_all(reg: ProcessRegistry) -> None:
    P = PropertyDef
    defs = [
        ProcessDef(
            "kin.free_motion", "Constant Velocity (Free Motion)", "1.0", "Kinematics",
            "Integrates position and orientation from the current linear and angular velocity. "
            "Use alone for uniform motion, or together with force processes.",
            "contribution", "per_object", [], ("linear_velocity", "angular_velocity"), (),
            "x(t+Δt) = x(t) + v Δt ;  q(t+Δt) = exp(½ ω Δt) ⊗ q(t)", _free_motion, priority=0),
        ProcessDef(
            "kin.constant_acceleration", "Constant Acceleration", "1.0", "Kinematics",
            "Applies the object's own Linear acceleration and Angular acceleration properties as constant "
            "accelerations.",
            "contribution", "per_object", [], ("linear_acceleration", "angular_acceleration"), (W_FORCE, W_TORQUE),
            "a += a_object ;  α += α_object", _constant_acceleration, priority=1),
        ProcessDef(
            "dyn.gravity", "Uniform Gravity", "1.0", "Dynamics",
            "Uniform gravitational field. Combined with the initial velocity it produces projectile motion.",
            "contribution", "per_object",
            [P("g", "Gravitational acceleration", V3, [0.0, 0.0, -9.80665], quantity="acceleration")],
            (), (W_FORCE,), "a += g", _gravity, priority=2),
        ProcessDef(
            "dyn.applied_force", "Applied Force (Newton's Second Law)", "1.0", "Dynamics",
            "Applies the object's Applied force (global frame, through the centre of mass) and optionally its "
            "Applied torque.",
            "contribution", "per_object",
            [P("include_torque", "Include applied torque", ValueType.BOOL, True)],
            ("force", "torque", "mass", "inertia"), (W_FORCE, W_TORQUE),
            "a += F / m ;  α += I_w⁻¹ τ   (I_w = R I Rᵀ about the COM)", _applied_force, requires_mass=True,
            priority=3, preconditions="Mass must be positive; torque requires a non-singular inertia tensor."),
        ProcessDef(
            "dyn.damping", "Viscous Damping", "1.0", "Dynamics",
            "Linear viscous damping force opposing the velocity, with optional angular damping torque.",
            "contribution", "per_object",
            [P("linear", "Linear damping coefficient", F, 0.5, quantity="damping", minimum=0.0),
             P("angular", "Angular damping coefficient", F, 0.0, minimum=0.0, description="c_r in N·m·s/rad")],
            ("linear_velocity", "angular_velocity", "mass", "inertia"), (W_FORCE, W_TORQUE),
            "F = −c v ;  τ = −c_r ω", _damping, requires_mass=True, priority=4),
        ProcessDef(
            "dyn.spring", "Spring-Damper (two bodies)", "1.0", "Dynamics",
            "Linear spring and damper acting along the line between the centres of mass of exactly two bodies. "
            "Global process: evaluated once per step for the pair.",
            "contribution", "global",
            [P("stiffness", "Stiffness k", F, 50.0, quantity="stiffness", minimum=0.0),
             P("rest_length", "Rest length L₀", F, 1.0, quantity="length", minimum=0.0),
             P("damping", "Damping c", F, 0.0, quantity="damping", minimum=0.0)],
            ("position", "linear_velocity", "mass"), (W_FORCE,),
            "F_a = [k(|d| − L₀) + c (v_b − v_a)·d̂] d̂ ,  F_b = −F_a ,  d = x_b − x_a", _spring,
            requires_mass=True, min_targets=2, max_targets=2, priority=5),
        ProcessDef(
            "kin.circular_motion", "Circular Motion (prescribed)", "1.0", "Kinematics",
            "Moves the object origin on a circle at constant angular speed. Orientation is unaffected.",
            "prescribed", "per_object",
            [P("center", "Centre", V3, [0.0, 0.0, 0.5], quantity="length"),
             P("radius", "Radius", F, 1.0, quantity="length", minimum=0.0),
             P("axis", "Normal axis", V3, [0.0, 0.0, 1.0]),
             P("angular_speed", "Angular speed Ω", F, 1.0, quantity="angular_velocity"),
             P("phase", "Phase φ", F, 0.0, quantity="angle")],
            (), (W_POSITION,), "p(t) = c + r[cos(φ+Ωt) e₁ + sin(φ+Ωt) e₂]", _circular, priority=10),
        ProcessDef(
            "kin.oscillation", "Harmonic Oscillation (prescribed)", "1.0", "Kinematics",
            "Sinusoidal oscillation of the object origin about its initial position.",
            "prescribed", "per_object",
            [P("direction", "Direction", V3, [0.0, 0.0, 1.0]),
             P("amplitude", "Amplitude A", F, 0.5, quantity="length"),
             P("frequency", "Frequency f", F, 0.5, quantity="frequency", minimum=0.0),
             P("phase", "Phase φ", F, 0.0, quantity="angle")],
            ("position",), (W_POSITION,), "p(t) = p₀ + A sin(2πf t + φ) d̂", _oscillation, priority=11),
        ProcessDef(
            "rot.fixed_axis", "Rotation About a Fixed Axis (prescribed)", "1.0", "Rotation",
            "Rotates the object about a global axis through its own origin at constant angular speed.",
            "prescribed", "per_object",
            [P("axis", "Axis (global)", V3, [0.0, 0.0, 1.0]),
             P("angular_speed", "Angular speed Ω", F, 1.0, quantity="angular_velocity")],
            ("orientation",), (W_ORIENTATION,), "q(t) = exp(½ Ω t â) ⊗ q₀ ;  ω = Ω â", _rotation_fixed_axis,
            priority=12),
        ProcessDef(
            "rot.local_axis", "Rotation About a Local Axis (prescribed)", "1.0", "Rotation",
            "Spins the object about an axis fixed in its own local frame at constant angular speed.",
            "prescribed", "per_object",
            [P("axis", "Axis (local)", V3, [0.0, 0.0, 1.0]),
             P("angular_speed", "Angular speed Ω", F, 1.0, quantity="angular_velocity")],
            ("orientation",), (W_ORIENTATION,), "q(t) = q₀ ⊗ exp(½ Ω t â_local) ;  ω = R(t) Ω â_local",
            _rotation_local_axis, priority=13),
    ]
    for d in defs:
        reg.register(d)
