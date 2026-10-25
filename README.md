# PSim Studio — Professional 3D Physics Simulation Studio

PSim Studio is a desktop 3D physics simulation studio. It is written in Python (PySide6 / Qt 6)
with a C++17 numerical core bound through pybind11.

This repository delivers **Phase 1** of the development plan: a working desktop application with
- a real OpenGL 3D viewport
- a searchable object library with real 3D thumbnails
- drag-and-drop placement
- an undoable, typed property system with custom properties
- coordinate axes, configurable grids and transform gizmos
- project save/load with autosave and recovery

The architecture already contains the interfaces that later phases build on (the simulation
timing contract, a unit registry, an object type registry and a plugin API).

![Main window](docs/screenshot_main.png)

---

## 1. Quick start

```bash
pip install -r requirements.txt          # PySide6, PyOpenGL, NumPy, pybind11, pytest, pytest-qt
python build_native.py                   # builds + tests the C++ core, installs it into psim/numerics/
python app.py                            # launch
python app.py examples/mechanical_demo.psim
python app.py --template primitives      # start from a built-in project template
```

Options: `--theme dark|light`, `--no-recovery`, `--reset-settings`.

**Windows.** You need Python ≥ 3.10 (64-bit), Visual Studio 2019+ ("Desktop development with
C++") and CMake ≥ 3.18. Run the same commands from a *Developer PowerShell*. `build_native.py`
copies `_psim_native*.pyd` next to the Python package.

**If the native core is not built**, the application still runs on an equivalent NumPy
implementation of every routine. The status bar then shows *NumPy fallback*, the console logs a
warning, and *Help ▸ About* names the active backend. The test suite checks that both
backends give the same results.

**Graphics requirement:** OpenGL 3.3 core profile, which any GPU driver from the last decade provides.

## 2. Tests

```bash
python -m pytest                 # Windows / Linux desktop
xvfb-run -a python -m pytest     # headless Linux (GUI tests need an OpenGL display)
ctest --test-dir native/build    # C++ unit tests (also run by build_native.py)
```

| Suite | What it verifies |
|---|---|
| `test_native.py` | C++ core ≡ NumPy fallback (quaternions, poses, matrices, ray casting, normals, feature edges, mass properties) |
| `test_objects.py` | every registered type builds valid, outward-oriented geometry; volumes vs. analytic formulas; inertia tensors (box, sphere, cylinder, thin plate); mass modes; registry validation; OBJ import |
| `test_properties.py` | typed coercion and validation, custom properties (keys, ranges, types), property paths, read-only rules |
| `test_units.py` | conversions, incompatible-dimension errors, custom units, preferences, per-property display units |
| `test_coordinates.py` | Euler convention, gimbal lock, global ↔ local poses through hierarchies, relative-frame representation |
| `test_editing.py` | unique persistent IDs (never reused), undo/redo of every edit type, group/ungroup, clipboard, align/distribute, parallel-axis combination |
| `test_project_io.py` | save/load round trip, rejection of malformed/unsupported/unsafe files, warnings, example projects |
| `test_update_rate.py` | Phase 3 timing contract: Δt independent of update rate, exact step counts, duration-boundary policies, no clock drift over 10⁶ steps, bounded catch-up, overrun reporting |
| `test_interaction_math.py` | camera projection/rays, picking, surface/plane placement, gizmo translate/rotate/scale with snapping |
| `test_gui.py` | app launch and menus, real 3D rendering (pixel checks, shading, shadows), drag-and-drop, click selection, gizmo drag + undo, inspector editing, display units, save/reopen, themes, camera follow modes |

Current result: **165 Python tests and the C++ test executable pass**. This was measured on
Linux with Mesa llvmpipe under Xvfb. Windows has not been tested yet; see §6.

## 3. What Phase 1 implements

**Application shell**
- Menu bar with all 14 requested menus. Physics, Processes, Coordinates, Plots and Simulation
  show their future features as *disabled "Planned: …" entries*. Nothing non-functional is
  presented as working.
- Main toolbar: New/Open/Save/Save As, Undo/Redo, Add Object, Select/Move/Rotate/Scale,
  global/local gizmo, coordinate axes, grid, snapping, work plane, properties, camera controls,
  full screen.
- Dockable, hideable and resizable panels. Named workspace layouts (save, restore, reset).
- Graphite dark and light themes, a vector SVG icon set and high-DPI support.
- Customizable keyboard shortcuts with conflict detection, and context menus in the viewport and outliner.

**3D viewport** (`psim/rendering`)
- OpenGL 3.3 core renderer:
  - GGX/PBR-style shading (metallic, roughness, opacity) with a key light, fill light and environment reflection
  - directional shadow mapping (PCF) plus a ground-shadow catcher
  - 4× MSAA
  - transparency sorting, wireframe mode and feature-edge rendering
- Camera:
  - orbit, pan and zoom-to-cursor
  - perspective/orthographic, front/back/left/right/top/bottom and isometric views
  - frame selected, fit scene, focus on origin
  - bookmarks
  - follow, look-at and track-local-frame modes (camera only; objects are never modified)
- Independent XY/YZ/XZ grids: colour, opacity, major spacing, subdivisions, extent, thickness,
  labels and origin emphasis. Lines are anti-aliased screen-space quads, so thickness works
  in a core profile.
- Global axes with configurable colours, length and thickness, plus labels, an origin marker and a view triad.
- Per-object overlays:
  - name, ID, position, orientation, velocity, acceleration, angular velocity, mass, force, torque and custom properties
  - local axes, bounding box and centre-of-mass symbol
  - per-object font size, precision and colour, and the project display units
- Selection highlight and hover, ray picking (native C++), and Ctrl/Shift multi-select.
- Transform gizmos: axis, plane and view-plane move; axis rotate; axis and uniform scale.
  They work in global or local orientation with grid, rotation and scale snapping, show a live
  readout, are a single undo step per drag, and Esc cancels.
- Drag-and-drop from the library with a ghost preview. The object is placed resting on the
  surface under the cursor (surface snapping) or on the chosen XY/YZ/XZ work plane, with grid snapping.

**Object library and types** (`psim/scene`, `psim/gui/object_library.py`)
- 29 registered types, each with real triangulated geometry and normals:
  - **Basic solids:** cube, cuboid, sphere, ellipsoid, cylinder, cone, frustum, torus, plane,
    disk, capsule, pyramid, prism, hemisphere, plus a wedge from the example plugin
  - **Mechanical:** spring, rope/cable, spur gear, I-beam, pulley
  - **Bodies:** point mass, rigid body (box, cylinder)
  - **Reference frames:** reference frame, coordinate marker
  - **Vectors & measurement:** line segment, vector arrow
  - **Groups:** assembly
  - **User-defined:** custom polygonal mesh (Wavefront OBJ import)
- Composite templates: dumbbell, wheel & axle, simple pendulum, spring-mass and gear pair.
  User templates come from *Objects ▸ Save as Template*.
- The thumbnails are software-rendered from the actual meshes. The library is searchable and has collapsible categories.
- Extension API: `ObjectRegistry.register(ObjectTypeDef(...))`. Plugins in `psim/plugins/`
  are loaded at start-up (see `example_wedge.py`). Project files can never register or run code.

**Objects, properties and editing**
- Unique, persistent IDs (`OBJ-000001`, …). They are never reused, survive renames and edits,
  and duplicates are rejected on load.
- Inspector sections:
  - **Identity, Geometry and Transform**, the transform shown relative to the global frame, the parent or any object
  - **Physical Properties:**
    - mass from density or mass, physical material presets
    - volume or area, centre of mass, inertia tensor and principal moments, computed by exact
      polyhedral integration of the faceted mesh in C++
    - initial velocities, accelerations, force and torque
    - derived momentum, angular momentum and kinetic energy
  - **Coordinate Systems:** parent, global and local pose, local axes
  - **Materials & Appearance**
  - **Labels & Display**
  - **Custom Properties:** float, int, bool, string, 2D/3D vector, with unit, description,
    default, read-only flag, precision, min/max validation and sorting
- Every value carries a **U/C/D/S** badge (user-controlled, calculated, derived, simulated). The inspector
  has a property filter, per-property display-unit selectors and multi-object editing.
- Scene hierarchy outliner:
  - drag-to-reparent with cycle protection, visibility and lock toggles, filtering
  - group/ungroup, duplicate, copy/cut/paste, delete, rename
  - align (min/centre/max) and distribute, hide/show/lock
- Undo/redo for every edit through before/after object snapshots, plus a history panel.
- Bottom panels: Console, Errors & Warnings, Object Table (sortable, filterable, CSV export) and
  Performance (frame time, draw calls, triangles, sparkline).
- Mass properties report: combined mass, combined centre of mass and inertia via the parallel-axis theorem.

**Units** (`psim/units`): SI canonical storage and dimension vectors (with an angle pseudo-dimension).
- About 100 units across all requested quantities.
- Project preferred units, per-property display units, and custom units with conversion
  factors. Incompatible conversions raise errors.

**Projects** (`psim/project`)
- Versioned JSON `.psim` format (documented in `serialization.py`).
- It stores objects, hierarchy, custom properties, mesh assets, units, grids/axes/snap/lighting,
  camera and bookmarks, simulation settings and the UI layout.
- Loading validates everything, and a failed load leaves the open project untouched. Saves are atomic.
- Autosave to a recovery file with a restore offer after a crash, recent files, project
  templates, OBJ import/export and CSV export.

## 4. Architecture

```
app.py                     launcher (OpenGL format, theme, CLI)
build_native.py            CMake configure/build/test/install of the C++ core
native/                    C++17 numerical core + pybind11 bindings + C++ tests
  include/psim/vecmath.hpp     Vec3 / Quat / Mat3 / Mat4, Euler (extrinsic XYZ) conventions
  include/psim/transforms.hpp  poses, TRS, look-at, perspective / orthographic
  src/mesh_ops.cpp             ray casting, normals, feature edges, solid & shell mass properties
psim/
  numerics/        facade: native module or NumPy fallback with an identical API
  core/            persistent ID allocator
  units/           dimensions, unit registry, display preferences
  properties/      typed PropertyDef, value origins, custom properties
  scene/           geometry generators, object registry, built-in types, SceneObject, Scene,
                   undoable SceneEditor, analysis
  rendering/       camera, shaders, GPU resources, renderer, grids/axes, gizmo, picking,
                   overlays, thumbnails, viewport widget, serializable view settings
  gui/             main window, actions/shortcuts, library, inspector, outliner, panels, dialogs, theme, icons
  project/         serialization, autosave/recovery, project templates
  simulation/      Phase 3 timing contract: SimulationSettings, SimulationClock, UpdateRateController
  plugins/         trusted application plugins (object types)
tests/             pytest suites (see §2)
examples/          sample projects
```

Key design decisions:

- **Single access path for values.** Property paths (`position`, `params.radius`,
  `custom.<key>`, `overlay.<name>`) are used by the inspector, the undo system, multi-editing
  and, later, the process engine. All writes are validated by the `PropertyDef` of the target.
- **Undo is state-based.** A command stores the serialized before/after state of the affected
  objects, which covers add, delete, edit, reparent, group and paste with one mechanism and
  restores IDs exactly.
- **Geometry is separate from physics.** Types generate meshes and the scene stores only
  parameters. Mass properties are derived from geometry, scale and density on demand. A `mass`
  value alone does not make an object a rigid body: types declare a mass model
  (solid/shell/point/none) and a `rigid_body` flag for the Phase 7 solver.
- **Rigid frames.** `position` and `orientation` are relative to the parent. `scale` is
  per-object geometry scale and is **not inherited**, so every object frame stays a rigid frame.
  This is what physics and relative kinematics will need.
- **Euler convention.** Extrinsic X then Y then Z, `R = Rz·Ry·Rx`, Z up, right-handed.
  Quaternions are stored as (w, x, y, z).

### Simulation timing contract (Phase 3 interface, implemented and tested now)

`psim/simulation/settings.py` and `clock.py` fix the semantics the engine must follow:

- **Δt** is the simulated-time increment per step. It is never derived from the update rate
  or the render rate.
- **Update rate (Hz)** is how many complete steps the real-time scheduler targets per
  wall-clock second. Simulated-time progression is `rate × Δt` simulated seconds per second
  (for example 100 Hz × 0.001 s = 0.1×).
- **The clock** computes time as `segment_start + k·Δt`, so there is no drift. A change of Δt
  starts a new segment and applies to subsequent steps.
- **Duration boundary policy:** final partial step (default, ends exactly at the duration),
  next full step, or last full step.
- **Real-time pacing** uses a monotonic clock with a maximum number of catch-up steps per tick.
  When the hardware falls behind, steps are *dropped from the schedule and reported as an
  overrun*. Δt is never enlarged. There is also an as-fast-as-possible mode.
- **Rendering FPS is a separate setting.** The viewport renders on demand or at a configurable
  continuous rate, and never touches model state (this is tested).

## 5. Phase 1 limitations (honest status)

- **Not implemented yet** (planned phases):
  - the running simulation engine and its controls (Phase 3)
  - the formula/process engine and developer-only process authoring with permissions (Phase 4)
  - named reference frames as first-class objects, frame-relative velocities/accelerations,
    and dimensional analysis inside formulas (Phase 5)
  - trajectories, plotting workspace and data recorder (Phase 6)
  - rigid-body dynamics, collisions and joints (Phase 7)
  - installer (Phase 8)

  The *Process Table*, *Data Inspector* and *Plot Preview* bottom panels are therefore not
  present yet. Trajectory settings are not shown in the inspector because there is no motion
  to record.
- Box (marquee) selection, measurement tools and a numeric "move relative to object" dialog are
  not implemented. Relative placement is done through the inspector's *Relative to* frame
  selector and surface snapping.
- Imported meshes are stored inside the project file (as JSON arrays) rather than as
  separate asset files. Very large meshes make projects large.
- When the theme is switched at runtime, icons on a few panel buttons keep the previous tint
  until restart. Menus, toolbar, library, inspector and outliner refresh immediately.
- The renderer issues one draw call per object (no instancing). It is comfortable for scenes of
  hundreds of objects but not tuned for tens of thousands.
- Developed and tested on Linux with software OpenGL (Mesa llvmpipe). The code is
  platform-neutral (Qt, CMake/MSVC-safe C++), but **it has not been run on Windows in this
  environment**.

## 6. Next steps

1. Run the test suite on Windows with a hardware GPU.
2. Phase 2 remainder: placement-relative dialog, marquee selection, trajectory settings model.
3. Phase 3: worker-thread engine built on `SimulationClock` / `UpdateRateController`, with
   state double-buffering and a staged update policy. Add simulation toolbar controls and a
   live rate/overrun display.
