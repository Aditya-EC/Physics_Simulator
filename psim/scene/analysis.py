"""Scene-level analysis helpers."""
from __future__ import annotations

import numpy as np

from .. import numerics
from .scene_manager import Scene


def combined_mass_properties(scene: Scene, ids: list[str]) -> dict:
    """Total mass, combined COM and inertia about it (global axes) via the parallel-axis theorem."""
    rows = []
    total = 0.0
    weighted = np.zeros(3)
    for oid in ids:
        o = scene.get(oid)
        if not o.typedef.physical:
            continue
        mp = o.mass_properties()
        m = mp["mass"]
        if not m:
            continue
        p, q = scene.world_pose(oid)
        r = np.asarray(numerics.quat_to_matrix(q))
        com_world = p + r @ mp["com"]  # COM is expressed in the (scaled) local frame
        i_world = r @ mp["inertia"] @ r.T
        rows.append((o, m, com_world, i_world, mp))
        total += m
        weighted += m * com_world
    if total <= 0:
        return {"rows": rows, "mass": 0.0, "com": np.zeros(3), "inertia": np.zeros((3, 3))}
    com = weighted / total
    inertia = np.zeros((3, 3))
    for _o, m, c, i_w, _mp in rows:
        d = c - com
        inertia += i_w + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return {"rows": rows, "mass": total, "com": com, "inertia": inertia}
