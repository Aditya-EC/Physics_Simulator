"""Numerical core facade.

Prefers the compiled C++ module (`_psim_native`, built with build_native.py).
If it cannot be imported, an equivalent NumPy implementation is used and
`BACKEND` reports "python" so the GUI and diagnostics can surface the
limitation. Set the environment variable PSIM_FORCE_PYTHON_NUMERICS=1 to
force the fallback (used by the cross-backend tests).
"""
from __future__ import annotations

import os

from . import _fallback

BACKEND = "python"
BACKEND_ERROR = ""
_impl = _fallback

if os.environ.get("PSIM_FORCE_PYTHON_NUMERICS") != "1":
    try:
        from . import _psim_native as _native  # type: ignore[attr-defined]

        _impl = _native
        BACKEND = "native"
    except ImportError as exc:  # pragma: no cover - depends on build state
        BACKEND_ERROR = str(exc)

FALLBACK = _fallback

quat_normalize = _impl.quat_normalize
quat_multiply = _impl.quat_multiply
quat_conjugate = _impl.quat_conjugate
quat_from_axis_angle = _impl.quat_from_axis_angle
quat_from_euler = _impl.quat_from_euler
quat_to_euler = _impl.quat_to_euler
quat_to_matrix = _impl.quat_to_matrix
matrix_to_quat = _impl.matrix_to_quat
quat_rotate = _impl.quat_rotate
pose_compose = _impl.pose_compose
pose_inverse = _impl.pose_inverse
trs_matrix = _impl.trs_matrix
look_at = _impl.look_at
perspective = _impl.perspective
orthographic = _impl.orthographic
transform_points = _impl.transform_points
ray_mesh_intersect = _impl.ray_mesh_intersect
compute_vertex_normals = _impl.compute_vertex_normals
feature_edges = _impl.feature_edges
mass_properties_solid = _impl.mass_properties_solid
mass_properties_shell = _impl.mass_properties_shell


def backend_description() -> str:
    if BACKEND == "native":
        return "Native C++ core (_psim_native)"
    reason = f": {BACKEND_ERROR}" if BACKEND_ERROR else ""
    return f"NumPy fallback (native core not loaded{reason})"
