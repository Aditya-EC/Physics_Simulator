"""Frame kinematics: representing vectors in a frame vs. motion *relative to* a moving frame.

Two different operations are easy to confuse:

* **Representation** (``express``): the same physical vector written with the
  components of another frame's axes, ``Rᵀ·v``. It only rotates; it does not
  depend on how the frame moves.
* **Relative motion** (``relative_motion``): what an observer *attached to* a
  moving frame measures. For a point at ``r = p − o_F`` (global vectors)::

      v_rel = Rᵀ (v − v_F − ω_F × r)
      a_rel = Rᵀ (a − a_F − α_F × r − ω_F × (ω_F × r) − 2 ω_F × (v − v_F − ω_F × r))

  i.e. the time derivatives of the point's coordinates in that frame. The
  subtracted terms are the transport velocity and the transport (origin,
  Euler, centripetal) and Coriolis accelerations.

All inputs and stored quantities are global (world) vectors in SI units;
results are components in the target frame's axes. Accelerations may be
unknown (``None``), e.g. before a simulation has computed them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import numerics


def _v(x) -> np.ndarray:
    return np.asarray(x, float).reshape(3)


@dataclass
class FrameState:
    """Kinematic state of a frame: its origin, orientation (axes = columns of ``rotation``) and rates, all global."""

    origin: np.ndarray
    quaternion: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    acceleration: np.ndarray | None = None
    angular_acceleration: np.ndarray | None = None

    @staticmethod
    def fixed(origin=(0.0, 0.0, 0.0), quaternion=(1.0, 0.0, 0.0, 0.0)) -> "FrameState":
        """A frame that does not move (the global frame when called without arguments)."""
        return FrameState(_v(origin), unit_quat(quaternion), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))

    @property
    def rotation(self) -> np.ndarray:
        return rotation_of(self.quaternion)

    @property
    def accelerations_known(self) -> bool:
        return self.acceleration is not None and self.angular_acceleration is not None


@dataclass
class RigidState:
    """Kinematic state of a rigid body: frame origin and orientation plus centre-of-mass motion (global)."""

    origin: np.ndarray
    quaternion: np.ndarray
    com: np.ndarray                  # global position of the centre of mass
    velocity: np.ndarray             # COM velocity
    angular_velocity: np.ndarray
    acceleration: np.ndarray | None = None          # COM acceleration
    angular_acceleration: np.ndarray | None = None

    @property
    def rotation(self) -> np.ndarray:
        return rotation_of(self.quaternion)

    def point_velocity(self, point) -> np.ndarray:
        return self.velocity + np.cross(self.angular_velocity, _v(point) - self.com)

    def point_acceleration(self, point) -> np.ndarray | None:
        if self.acceleration is None or self.angular_acceleration is None:
            return None
        r = _v(point) - self.com
        w = self.angular_velocity
        return self.acceleration + np.cross(self.angular_acceleration, r) + np.cross(w, np.cross(w, r))

    def frame(self) -> FrameState:
        """The body's own (local) frame, located at its origin."""
        return FrameState(self.origin.copy(), unit_quat(self.quaternion), self.point_velocity(self.origin),
                          self.angular_velocity.copy(), self.point_acceleration(self.origin),
                          None if self.angular_acceleration is None else self.angular_acceleration.copy())

    def attached(self, offset, offset_quaternion, follow_rotation: bool = True) -> FrameState:
        """A frame whose origin is the body point at ``offset`` (body axes) from the body origin.

        With ``follow_rotation`` the axes are the body axes rotated by ``offset_quaternion``;
        otherwise the axes keep the fixed global orientation ``offset_quaternion`` (a translating frame).
        """
        origin = self.origin + self.rotation @ _v(offset)
        q_off = unit_quat(offset_quaternion)
        if follow_rotation:
            q = unit_quat(numerics.quat_multiply(unit_quat(self.quaternion), q_off))
            w = self.angular_velocity.copy()
            alpha = None if self.angular_acceleration is None else self.angular_acceleration.copy()
        else:
            q, w, alpha = q_off, np.zeros(3), np.zeros(3)
        return FrameState(origin, q, self.point_velocity(origin), w, self.point_acceleration(origin), alpha)

    @staticmethod
    def at_rest(origin, quaternion, com=None) -> "RigidState":
        o = _v(origin)
        return RigidState(o, unit_quat(quaternion), o.copy() if com is None else _v(com), np.zeros(3), np.zeros(3),
                          np.zeros(3), np.zeros(3))


@dataclass
class RelativeMotion:
    """Motion of a point seen from a frame. Every vector is given in the frame's components."""

    position: np.ndarray
    velocity_components: np.ndarray           # Rᵀ v: absolute velocity, re-expressed (representation only)
    velocity_relative: np.ndarray             # observed by an observer moving with the frame
    transport_velocity: np.ndarray            # v_F + ω_F × r
    acceleration_components: np.ndarray | None = None
    acceleration_relative: np.ndarray | None = None
    origin_acceleration: np.ndarray | None = None      # a_F
    euler_acceleration: np.ndarray | None = None       # α_F × r
    centripetal_acceleration: np.ndarray | None = None  # ω_F × (ω_F × r)
    coriolis_acceleration: np.ndarray | None = None     # 2 ω_F × v_rel


@dataclass
class RelativeRotation:
    orientation: np.ndarray                  # quaternion of the body axes in the frame
    angular_velocity_components: np.ndarray  # Rᵀ ω
    angular_velocity_relative: np.ndarray    # Rᵀ (ω − ω_F)
    angular_acceleration_relative: np.ndarray | None = None  # Rᵀ (α − α_F − ω_F × ω)


def unit_quat(quaternion) -> np.ndarray:
    q = np.asarray(quaternion, float).reshape(4)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("Orientation quaternion must be non-zero")
    return q / n


def rotation_of(quaternion) -> np.ndarray:
    return np.asarray(numerics.quat_to_matrix(unit_quat(quaternion)), float)


def express(vector, frame: FrameState) -> np.ndarray:
    """Components of a global vector in the frame's axes (representation change, rotation only)."""
    return frame.rotation.T @ _v(vector)


def unexpress(components, frame: FrameState) -> np.ndarray:
    """Global vector from components in the frame's axes."""
    return frame.rotation @ _v(components)


def point_in_frame(point, frame: FrameState) -> np.ndarray:
    return frame.rotation.T @ (_v(point) - frame.origin)


def point_from_frame(coords, frame: FrameState) -> np.ndarray:
    return frame.origin + frame.rotation @ _v(coords)


def relative_motion(point, velocity, acceleration, frame: FrameState) -> RelativeMotion:
    rt = frame.rotation.T
    r = _v(point) - frame.origin
    w = frame.angular_velocity
    v = _v(velocity)
    transport = frame.velocity + np.cross(w, r)
    v_rel = v - transport
    out = RelativeMotion(rt @ r, rt @ v, rt @ v_rel, rt @ transport)
    if acceleration is not None and frame.accelerations_known:
        a = _v(acceleration)
        euler = np.cross(frame.angular_acceleration, r)
        centripetal = np.cross(w, np.cross(w, r))
        coriolis = 2.0 * np.cross(w, v_rel)
        out.acceleration_components = rt @ a
        out.acceleration_relative = rt @ (a - frame.acceleration - euler - centripetal - coriolis)
        out.origin_acceleration = rt @ frame.acceleration
        out.euler_acceleration = rt @ euler
        out.centripetal_acceleration = rt @ centripetal
        out.coriolis_acceleration = rt @ coriolis
    return out


def relative_rotation(quaternion, angular_velocity, angular_acceleration, frame: FrameState) -> RelativeRotation:
    rt = frame.rotation.T
    w = _v(angular_velocity)
    q_rel = numerics.quat_multiply(numerics.quat_conjugate(unit_quat(frame.quaternion)), unit_quat(quaternion))
    out = RelativeRotation(unit_quat(q_rel), rt @ w, rt @ (w - frame.angular_velocity))
    if angular_acceleration is not None and frame.accelerations_known:
        out.angular_acceleration_relative = rt @ (_v(angular_acceleration) - frame.angular_acceleration
                                                  - np.cross(frame.angular_velocity, w))
    return out
