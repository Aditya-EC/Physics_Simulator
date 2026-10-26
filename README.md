# PSim Studio — Professional 3D Physics Simulation Studio

PSim Studio is a desktop 3D physics simulation studio. It is written in Python (PySide6 / Qt 6)
with a C++17 numerical core bound through pybind11.

Implemented so far:

- **Phase 1**, a working desktop application:
  - a real OpenGL 3D viewport
  - a searchable object library with real 3D thumbnails
  - drag-and-drop placement
  - an undoable, typed property system with custom properties
  - coordinate axes, configurable grids and transform gizmos
  - project save/load with autosave and recovery
- **Phase 3**, the simulation engine:
  - a fixed Δt that is independent of the real-time update rate
  - a worker thread with a monotonic scheduler, bounded catch-up and overrun reporting
  - as-fast-as-possible mode
  - explicit Euler, semi-implicit Euler and RK4 integrators
  - built-in kinematics and dynamics processes with read/write declarations and conflict detection
  - Start/Pause/Resume/Stop/Reset/Single-Step, breakpoints and live parameter changes
- **Phase 4**, the formula and process engine:
  - a safe formula language (own parser and compiler, no `eval`) with static type and
    dimensional analysis and unit literals such as `9.81[m/s^2]`
  - formula-defined contribution, prescribed and discrete *update* processes, with
    dependency ordering, project-wide global parameters and custom-property outputs
  - developer-only authoring: a passphrase-protected developer role, a sealed process registry
    that enforces it, and Ed25519-signed process packages

![Main window](docs/screenshot_main.png)

---

## 1. Quick start

```bash
pip install -r requirements.txt          # PySide6, PyOpenGL, NumPy, pybind11, pytest, pytest-qt
python build_native.py                   # builds + tests the C++ core, installs it into psim/numerics/
python app.py                            # launch
python app.py examples/mechanical_demo.psim
python app.py --template simulation      # start from a built-in project template (also: primitives, mechanical)
python app.py examples/simulation_demo.psim   # then press F5 (Start)
python app.py examples/formula_demo.psim      # formula processes, global parameters, custom outputs
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
| `test_simulation.py` | §9.9: step counts, Δt per step, duration-boundary policies, Δt changes, bitwise determinism, independence from render FPS **and** pacing mode, real-time rate (100 Hz × 0.001 s ⇒ 0.1×), unachievable-rate reporting with Δt unchanged, pause/step/breakpoints, live changes, instability detection, integrator regressions (closed-form Euler errors, RK4 4th-order convergence, symplectic energy behaviour) |
| `test_processes.py` | every built-in process against analytic solutions (constant velocity/acceleration, F = ma, torque, damping decay, projectile range, circle, oscillation, rotations, spring period and momentum, COM offsets), conflict and eligibility diagnostics, summation, targets by type/tag, undoable assignments |
| `test_formulas.py` | formula parser (precedence, unit literals, error positions), type and dimension checking, runtime semantics and domain errors, aggregates, scripts (`let`/`require`/targets), test-case runner |
| `test_formula_processes.py` | library processes in the engine against analytic results (discrete update rules ≡ semi-implicit Euler bit-for-bit, N-body circular orbit, drag terminal velocity, prescribed Lissajous with numerical rates, kinetic-energy output), required globals, update dependency cycles, missing writers and references, error policies; permissions (sealed registry, wrong passphrase, forged token), signing, tamper and untrusted-key rejection, version rules, withdrawal, developer CLI, projects that try to embed formulas |
| `test_developer_gui.py` | developer-mode unlock, process editor (refuses without token, validate with line highlighting, tests, sign & publish), global-parameters dialog with unit conversion |
| `test_simulation_session.py` | session semantics (run → stop → undo → reset), exact reset, live edits applied at the next step, invalid plans, single step, assignment round-trip, v1 project compatibility, GUI toolbar/inspector integration |

Current result: **322 Python tests and the C++ test executable pass** with the native core.
With the NumPy fallback, 291 pass and 31 native-comparison tests are skipped. This was measured on
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

## 3b. What Phase 3 implements (simulation engine)

**Controls.**
- A **Simulation toolbar** has Start/Resume (F5), Pause (F6), Stop (Shift+F5), Reset
  (Ctrl+F5), Single Step (F10) and Run as fast as possible.
- Separate **Δt**, **Update rate (Hz)** and **Duration** fields sit alongside a live read-out of
  simulated time, completed steps, actual/requested rate, simulation speed (simulated s per
  real s), state, and a *behind schedule* warning with the dropped-step count.
- **Simulation ▸ Simulation Settings** adds:
  - the duration-boundary policy and the integrator
  - the maximum catch-up steps and the render rate
  - breakpoints at simulated times
  - the derived quantities: expected step count and rate × Δt
- The Performance panel and the viewport HUD show the same timing figures.

**Engine** (`psim/simulation`).
- A deterministic step loop implements §9.4: shared previous state, plan-ordered evaluation,
  atomic state replacement, clock advance by Δt, and the step counter.
- Integrators: explicit Euler, semi-implicit (symplectic) Euler and RK4. Processes are
  evaluated at each solver stage.
- Hot kernels run in C++: linear integration, the quaternion exponential map, and batched
  rotation of vectors and inertia tensors.
- The translational state is the centre of mass. Prescribed motion is defined for the object origin.
- Non-finite or exploding values stop the run with a diagnostic.

**Runner.**
- A worker thread with its own monotonic scheduler, separate from rendering.
- Commands are queued and applied between steps, so changes to Δt, rate, duration, pacing,
  object properties or process parameters never affect a half-finished step.
- Snapshots go to the GUI at the render rate and nothing more often, so rendering FPS never
  alters Δt or the number of steps.
- If the engine falls behind, it runs at most N catch-up steps per tick and reports the rest as
  dropped. Δt is never changed.
- On Windows the runner requests 1 ms timer resolution during a run.

**Processes** (`psim/processes`).
- Approved process definitions carry an ID, version, category, description, formula,
  parameters with units, reads/writes, eligibility, mode (per-object or global), preconditions
  and error handling.
- Built-ins:
  - **Kinematics:** constant velocity, constant acceleration, prescribed circular motion,
    prescribed harmonic oscillation
  - **Rotation:** about a fixed global axis, about an object-local axis
  - **Dynamics:** uniform gravity (projectiles), applied force and torque (Newton's second law
    with world-frame inertia), viscous damping, and a two-body spring-damper (a global process)
- Assignments target objects by ID, type or tag. Their parameters are editable, they are
  undoable, and they are saved in the project (file format v2, which still reads v1).
- A **Processes** panel (the Process Table) lists assignments in evaluation order with enable
  toggles. It provides validation, a **dependency and diagnostics viewer** (reads/writes per
  process and writers per object) and process documentation. The inspector has an **Assigned
  Processes** section.
- Conflict rules:
  - force and torque contributions are summed
  - two prescribed writers of the same quantity are a conflict
  - a per-object force on a prescribed body is a conflict
  - a global (multi-body) force on a prescribed *driven anchor* is ignored with a warning
- Other diagnostics: missing references, ineligible targets and invalid parameters are
  reported before the run, and a simulated object inside a simulated assembly is rejected.

**Session semantics.**
- During a session, edits apply live, undo is disabled, and structural edits are blocked.
- *Stop* records the whole run as one undoable step.
- *Reset* restores the exact state captured at Start.
- Inspector values owned by the running simulation show the **S** badge. The current total
  acceleration is shown live in the inspector and labels.

## 3c. What Phase 4 implements (formulas, processes, permissions)

**Formula language** (`psim/processes/formula_*.py`, `dimensional_analysis.py`).
- Hand-written lexer and Pratt parser to an AST, a static checker, and a compiler to Python
  closures. Nothing is ever passed to `eval`/`exec`; only whitelisted functions, attributes and
  constants exist. Sources are limited to 10 000 characters and nesting depth is bounded.
- Types: number, vec2, vec3, mat3, quaternion, bool, object, object set, string. Every number
  carries an SI dimension. `+`, `-`, comparisons and function arguments must agree
  dimensionally, so `self.position + self.velocity` is rejected *before* the run, with line,
  column and a caret. Literal `0` is compatible with any dimension.
- Unit literals (`2.5[km/h]`, `9.81[m/s^2]`, `30[deg]`) are converted to SI at compile time.
  Angles are dimensionless inside formulas (radians).
- Built-ins: vector/matrix/quaternion algebra, `norm`, `normalize`, `dot`, `cross`,
  trigonometry, `clamp`, `lerp`, `if(c, a, b)`, frame transforms (`to_local`, `to_global`, `point_to_local`, `point_to_global`, `rotate`),
  aggregates `sum_of / mean_of / min_of / max_of / count` over `others`, `targets`,
  `tagged("…")` or `of_type("…")`, and `object("OBJ-…")` references.
- Domain errors (division by zero, `sqrt` of a negative, normalising a zero vector, non-finite
  results) raise a diagnostic naming the process, the object and the expression.

**Scripts and processes** (`formula_process.py`).
- A script is a list of `let name = expr`, `require cond : "message"` and `target = expr`
  lines. The allowed targets depend on the kind:
  - *contribution*: `force`, `torque`, `acceleration`, `angular_acceleration` (summed with
    other contributions)
  - *prescribed*: `position` (+ optional `velocity`), `orientation` (+ optional
    `angular_velocity`). If the rate is omitted it is computed by a central difference.
  - *update*: discrete-time assignments to `position`, `velocity`, `orientation`,
    `angular_velocity` or `custom.<key>`, applied after integration. They read the state at
    the start of the step, or the value already updated this step with `updated(self.velocity)`.
- Per-object processes see `self`; global processes see `targets[i]` and write
  `targets[i].force`, etc.
- Specs declare parameters with quantities and ranges, required **global parameters** (name,
  type, quantity), required custom properties, `after` dependencies, an error policy (`stop`
  the run, or `skip` this object this step and report it) and embedded **test cases**.
- **Process library** (`psim/processes/library/*.json`, loaded at start-up and only if every
  test passes): quadratic drag, wind drag (global `air_density` and `wind`), N-body gravity,
  periodic force, tether (global spring-damper), Lissajous path (prescribed), discrete Euler
  velocity/position/angular-velocity/orientation updates and a kinetic-energy monitor that
  writes `custom.kinetic_energy`.

**Scheduler additions.**
- Update processes are topologically ordered from `after` and `updated()` references.
  Cycles, `updated()` of a quantity no process updates, missing object references and missing
  or wrongly-typed globals are errors before the run.
- An update process that writes velocity or position cannot be combined with forces on the
  same object (the integrator would also move it). Velocity updated without position is
  reported as a warning.

**Global parameters.** *Processes ▸ Global Parameters…* edits project-wide named values
(float or vec3, with a quantity, shown in display units). They are undoable, saved in the
project and applied live during a run.

**Permissions and signing** (`permissions.py`, `packages.py`, `process_registry.py`, `devtools.py`).
- The default role is *user*: users select, configure and run approved processes only.
  Project files contain only process IDs and parameter values. An assignment with any extra
  field (e.g. an embedded script) is rejected on load.
- Developer credentials are created **outside the GUI**:
  `python -m psim.devtools init-developer --author "Name"`. This stores a PBKDF2-SHA256
  passphrase verifier and an Ed25519 signing key encrypted with the passphrase in the user's
  data directory. Other commands: `show-public-key`, `trust-key`, `list-trusted`, `verify`.
- *Tools ▸ Developer Mode…* unlocks a session token. Only then does the **Developer** menu
  appear (New Formula Process, Edit Formula Process (new version), Withdraw Published Process).
- The process registry is **sealed** after start-up. `register`, `publish` and `withdraw`
  verify the token themselves, so the GUI is not the enforcement point. Publishing requires
  validation, passing tests and a higher version, cannot replace built-in or library
  processes, and writes a signed `.psimproc` package.
- At start-up only packages signed by a trusted key with a valid signature are loaded. Edited
  or foreign packages are reported and ignored.
- The **Process Editor** has metadata, a parameter table, globals/custom/test JSON, a script
  editor with a language reference, *Validate* (highlights failing lines), *Run Tests*, drafts
  and *Sign & Publish*.

**Limits of this protection (honest status).** This model stops users of the application
from defining or altering processes through the application and detects tampered packages.
It does **not** protect against someone who can modify the Python source of the installation
or run arbitrary code as the same OS user. Python code can be read and patched, and no
client-side scheme prevents reverse engineering. A stronger guarantee would need
an OS-level install location writable only by administrators, which is a Phase 8
packaging concern.

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
  processes/       process definitions, sealed registry, built-in processes, scheduler (plan + diagnostics),
                   formula parser / dimensional checker / compiler, formula process specs + test runner,
                   permissions (developer role, Ed25519), signed packages, global parameters, library/*.json
  simulation/      settings, clock / rate controller, state, integrators, engine, threaded runner,
                   world (scene view for formulas, custom outputs)
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

### Simulation timing contract

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

## 5. Limitations (honest status)

- **Not implemented yet** (planned phases):
  - named reference frames as first-class objects and frame-relative
    velocities/accelerations (Phase 5)
  - trajectories, plotting workspace and data recorder (Phase 6)
  - rigid-body dynamics, collisions and joints (Phase 7)
  - installer (Phase 8)

  The *Data Inspector* and *Plot Preview* bottom panels are therefore not present yet, and
  trajectories are not recorded yet (Phase 6).
- Simulation physics limits for this phase:
  - Rotational dynamics uses α = I_w⁻¹·τ without the gyroscopic ω × Iω term, which belongs to
    the Phase 7 rigid-body solver.
  - There are no collisions, so objects pass through each other and the ground.
  - Spring and rope *geometry* does not deform with the spring process.
  - The engine is vectorised NumPy plus C++ kernels: about 0.5 ms (semi-implicit) or 1.4 ms
    (RK4) per step for a 7-body scene on this machine. That supports real-time rates of a
    few hundred Hz for small scenes; a fully native step loop is planned with Phase 7.
  - Under software OpenGL, rendering competes with the engine thread for the Python GIL. The
    overrun detection then correctly reports reduced achieved rates.
- Formula processes limits:
  - They are evaluated as compiled Python closures per object and per solver stage, not in
    C++. They are therefore slower than the vectorised built-ins: the 4-body formula demo
    takes about 1.4 ms per RK4 step. Aggregates over many objects scale as O(n²).
  - Formulas cannot create objects, loop, or define functions. Conditionals are expressions
    (`if(c, a, b)`).
  - Dimensions are checked; units of *custom properties* are taken from their declared quantity.
  - The process editor uses JSON text fields for globals, custom properties and test cases.
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
2. Phase 5: named reference frames and frame-relative kinematics.
3. Phase 6: data recorder (sampling every N steps), trajectories and the plotting workspace.
