"""Placing objects relative to a reference: another object, a named frame, or the global frame.

The primary object is put at ``offset`` from the reference origin — along the reference's axes or the global
axes — and optionally given an orientation relative to the reference. Every other selected object moves with it
as one rigid group (its pose relative to the primary is kept), and children follow their parents.
"""
from __future__ import annotations

import numpy as np

from .. import numerics


class PlacementError(ValueError):
    pass


AXES = ("reference", "global")


def moved_ids(scene, ids) -> set[str]:
    """The objects that move: the selection and all their descendants."""
    out = set()
    for i in ids:
        out.add(i)
        out.update(scene.descendants(i))
    return out


def check_reference(scene, ids, ref: str | None) -> None:
    """Raise PlacementError if `ref` cannot serve as the reference for moving `ids`."""
    if not ids:
        raise PlacementError("Select the objects to place")
    missing = [i for i in ids if i not in scene.objects]
    if missing:
        raise PlacementError(f"Missing object '{missing[0]}'")
    locked = [scene.get(i).name for i in ids if scene.is_effectively_locked(i)]
    if locked:
        raise PlacementError(f"'{locked[0]}' is locked; unlock it to move it")
    if not ref:
        return
    moved = moved_ids(scene, ids)
    if ref.startswith("FRM-"):
        f = scene.frame(ref)
        if f is None:
            raise PlacementError(f"Missing reference frame '{ref}'")
        if f.parent in moved:
            raise PlacementError(f"The frame '{f.name}' is attached to an object that is being moved")
        return
    if ref not in scene.objects:
        raise PlacementError(f"Missing reference object '{ref}'")
    if ref in moved:
        raise PlacementError("An object cannot be placed relative to itself or to one of the objects being moved")


def current_offset(scene, oid: str, ref: str | None, axes: str = "reference") -> tuple[np.ndarray, np.ndarray]:
    """(offset, relative orientation) of `oid` with respect to `ref`, matching place()'s arguments."""
    rp, rq = scene.reference_pose(ref)
    wp, wq = scene.world_pose(oid)
    rel_q = np.asarray(numerics.quat_multiply(numerics.quat_conjugate(rq), wq), float)
    d = np.asarray(wp, float) - np.asarray(rp, float)
    if axes == "reference":
        d = np.asarray(numerics.quat_rotate(numerics.quat_conjugate(rq), d), float)
    return d, np.asarray(numerics.quat_normalize(rel_q), float)


def place(scene, ids, primary: str, ref: str | None, offset, rel_orientation=None,
          axes: str = "reference") -> dict[str, tuple]:
    """New global poses {id: (position, quaternion, None)} for the root objects of the selection `ids`
    (the format of SceneEditor.set_world_poses). `rel_orientation` None keeps the current orientation."""
    ids = list(dict.fromkeys(ids))
    check_reference(scene, ids, ref)
    if primary not in ids:
        raise PlacementError("The primary object must be one of the selected objects")
    if axes not in AXES:
        raise PlacementError("axes must be 'reference' or 'global'")
    offset = np.asarray(offset, float)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise PlacementError("The offset needs three finite values")
    rp, rq = scene.reference_pose(ref)
    target_p = np.asarray(rp, float) + (np.asarray(numerics.quat_rotate(rq, offset), float) if axes == "reference"
                                        else offset)
    cur_p, cur_q = scene.world_pose(primary)
    if rel_orientation is None:
        target_q = np.asarray(cur_q, float)
    else:
        rel = np.asarray(rel_orientation, float)
        if rel.shape != (4,) or not np.all(np.isfinite(rel)) or np.linalg.norm(rel) < 1e-12:
            raise PlacementError("The relative orientation must be a non-zero quaternion")
        target_q = np.asarray(numerics.quat_normalize(numerics.quat_multiply(rq, rel)), float)
    # rigid motion T taking the primary from its current pose to the target: T = target ∘ current⁻¹
    ip, iq = numerics.pose_inverse(cur_p, cur_q)
    tp, tq = numerics.pose_compose(target_p, target_q, ip, iq)
    chosen = set(ids)
    roots = [i for i in ids if not any(a in chosen for a in scene.ancestors(i))]
    out = {}
    for i in roots:
        p, q = scene.world_pose(i)
        np_, nq = numerics.pose_compose(tp, tq, p, q)
        out[i] = (np.asarray(np_, float), np.asarray(nq, float), None)
    return out
