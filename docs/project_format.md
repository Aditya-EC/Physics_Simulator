# PSim Studio file formats

PSim Studio uses two JSON file types. Both are plain data: loading them never executes code.
Every field is type-checked and validated, and anything the application does not understand is
rejected with a message rather than ignored silently.

| Extension | Content | Written by |
|---|---|---|
| `.psim` | a project: scene, processes, settings, plots | *File ▸ Save*, autosave (recovery copy) |
| `.psimstate` | a simulation state to continue a run from | *Simulation ▸ Save Simulation State*, `psim-batch --save-state` |

Files are written atomically (temporary file + rename), so a crash never leaves a half-written file.

## Project file (`.psim`)

```jsonc
{
  "format": "psim-project",
  "format_version": 2,                 // files from newer versions are refused with a clear message
  "app_version": "0.1.0",
  "metadata": {"title": "…", "created": "2026-10-26T14:03:34+00:00", "modified": "…"},
  "id_counter": 13,                     // next object number: IDs (OBJ-000001 …) are never reused
  "objects": [ … ],                     // scene order; parents are resolved after loading
  "processes": [ … ], "process_counter": 4,
  "global_parameters": [ … ],
  "reference_frames": [ … ], "frame_counter": 2,
  "joints": [ … ], "joint_counter": 6,
  "measurements": [ … ], "measurement_counter": 3,
  "assets": {"meshes": {"<sha256>": {"name": "…", "vertices": [[x, y, z] …], "indices": [[i, j, k] …]}}},
  "units": {"preferred": {"length": "m", …}, "custom_units": [ … ], "property_units": { … }},
  "view": { … },                        // grids, axes, snapping, environment, camera, bookmarks
  "simulation": { … },                  // see "Simulation settings"
  "ui": {"layout": "<base64 Qt window state>"},
  "plots": [ … ]                        // plot definitions (recorded data is not stored)
}
```

Version 1 files (Phase 1, without processes) still load. Optional keys may be missing; they then
take their defaults.

### Objects

```jsonc
{
  "id": "OBJ-000004", "type": "body.rigid_box", "name": "Crate 1", "parent": null,
  "position": [x, y, z],               // in the parent's frame (global for root objects)
  "orientation": [w, x, y, z],         // unit quaternion, relative to the parent
  "scale": [sx, sy, sz],               // geometry scale; not inherited by children
  "params": {"length": 0.5, …},        // type parameters, validated by the object type
  "color": [r, g, b], "opacity": 1.0, "material": "Matte", …,
  "mass_mode": "density" | "mass", "density": 1000.0, "mass_input": 1.0,
  "linear_velocity": [ … ], "angular_velocity": [ … ], "force": [ … ], "torque": [ … ], …,
  "tags": ["…"], "visible": true, "locked": false,
  "overlays": {"name": true, …}, "trajectory": { … }, "label_units": { … }, "arrow_scales": { … },
  "physics": {"collider": "auto", "restitution": 0.75, …},   // only non-default values
                                       // collider: auto | none | sphere | box | capsule | plane | convex | convex_parts
  "custom": [{"key": "drag_cd", "name": "Drag coefficient", "type": "float", "value": 0.47, …}]
}
```

Unknown object types (e.g. a plugin that is not installed) make the load fail with a message.
A missing mesh asset gives a warning and a placeholder box.

### Process assignments

```jsonc
{"uid": "PRC-0001", "process": "dyn.gravity",
 "targets": {"ids": ["OBJ-000002"], "types": [], "tags": []},
 "params": {"g": [0, 0, -9.80665]}, "enabled": true, "label": "Earth gravity"}
```

A project only *refers* to processes by ID. Formula processes live in the application's process
library; a project cannot contain or define one. An assignment of a process that is not installed
is kept, but disabled, with a warning.

### Reference frames, joints and measurements

```jsonc
{"id": "FRM-0001", "name": "Turntable", "parent": "OBJ-000003" | null, "position": [ … ],
 "orientation": [w, x, y, z], "follow_rotation": true, "visible": true, "show_label": true,
 "axis_length": 0.6, "color": [ … ], "description": ""}

{"id": "JNT-0001", "name": "Hinge", "type": "fixed" | "ball" | "revolute" | "prismatic" | "distance",
 "body_a": "OBJ-000008", "body_b": null,             // null = the fixed world
 "anchor_a": [ … ], "anchor_b": [ … ], "axis_a": [ … ], "axis_b": [ … ],  // in each body's local frame
 "ref_orientation": [w, x, y, z], "limit_enabled": false, "lower": 0.0, "upper": 0.0,
 "motor_enabled": false, "motor_speed": 0.0, "motor_max": 0.0, "length": 1.0, "rope": false,
 "collide_connected": false, "enabled": true, "description": ""}

{"id": "MSR-0001", "name": "Gap", "kind": "distance" | "angle",
 "points": [{"object": "OBJ-000002", "at": "origin" | "com"},      // 2 points, or 3 with the vertex second
            {"object": "OBJ-000003", "at": "point", "local": [x, y, z]}],  // fixed on the object: geometry
                                                                         // coordinates before scale
 "visible": true}
```

References to missing objects produce warnings. Frames are kept detached, joints are kept and
reported by validation, and measurements of missing objects are removed.

### Simulation settings

```jsonc
{"delta_time": 0.005, "update_rate_hz": 200.0, "duration": 20.0, "render_fps": 60.0,
 "pacing": "real_time" | "afap", "duration_policy": "final_partial_step" | "next_full_step" | "last_full_step",
 "max_catch_up_steps": 10, "integrator": "semi_implicit_euler" | "explicit_euler" | "midpoint" |
 "velocity_verlet" | "rk4" | "dopri5", "adaptive_tolerance": 1e-8, "gyroscopic": true,
 "physics_enabled": true, "solver_iterations": 20, "position_iterations": 8, "baumgarte": 1.0,
 "contact_slop": 0.002, "restitution_threshold": 0.2, "continuous_collision": true,
 "record_enabled": true, "record_every_n_steps": 5, "record_max_mb": 256.0,
 "record_overflow": "decimate" | "window" | "stop", "record_process_outputs": true,
 "breakpoints": [1.5, 3.0],
 "stop_conditions": [{"name": "Ground", "expression": "object(\"OBJ-000002\").position.z < 0[m]",
                      "action": "pause" | "stop", "enabled": true}]}
```

Stop-condition expressions are formulas. They are parsed, type-checked and dimension-checked when
a run is planned; they are never evaluated as code.

## Simulation state (`.psimstate`)

```jsonc
{
  "format": "psim-simulation-state", "version": 1,
  "time": 0.75,
  "clock": {"segment_start": 0.0, "segment_steps": 150, "partial_time": 0.0, "steps": 150, "delta_time": 0.005},
  "integrator": "rk4",
  "bodies": {"OBJ-000002": {"x": [ … ], "q": [w, x, y, z], "v": [ … ], "w": [ … ], "acc": [ … ], "alpha": [ … ]}, …},
  "custom": {"OBJ-000002": {"energy": 12.5}},       // custom properties written by update processes
  "physics": {"keys": [ … ], "touching": [[a, b], …], "last_dv": [ … ], "last_dw": [ … ]} | null,
  "conditions": {"Ground": false},                  // so a true condition does not fire again
  "saved": {"objects": 3, "application": "PSim Studio"}
}
```

`x` is the centre of mass (global), `v` its velocity, `q` the orientation and `w` the angular
velocity (global). A state can only be restored into a run that simulates exactly the same set of
objects. Continuing from it is bit-for-bit identical to the uninterrupted run (same machine and
backend). Restoring discards recorded samples at or after the restored time.
