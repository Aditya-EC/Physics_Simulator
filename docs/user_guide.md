# PSim Studio user guide

This guide walks through the application in the order a typical project is built: scene, physical
properties, processes, simulation, measurements and results. Every menu command named here exists
in the application. *Help ▸ Controls & Shortcuts* lists the keyboard and mouse bindings, and they can
be changed in *Tools ▸ Keyboard Shortcuts*.

## 1. Starting

- **Installed:** start *PSim Studio* from the Start menu or application launcher, or run
  `psim-studio [project.psim]`.
- **From a source checkout:** run `python app.py [project.psim]`.

Useful options:
- `--template NAME` starts from a built-in template: empty, primitives, mechanical, simulation,
  formulas, frames, recording or physics.
- `--theme dark|light` chooses the theme.
- `--reset-settings` forgets the window layout and preferences.

The example projects (in `examples/` or the installed *Example projects* folder) are a good tour.
Open one and press **F5** to run it:

| Project | Shows |
|---|---|
| `simulation_demo.psim` | built-in kinematics and dynamics processes |
| `formula_demo.psim` | formula processes, global parameters, custom outputs |
| `frames_demo.psim` | reference frames and the Relative Motion panel |
| `recording_demo.psim` | trajectories and ready-made plots |
| `physics_demo.psim` | collisions, a stack, a hinge, a rope chain and a motorised slider |
| `measurements_demo.psim` | live distance/angle measurements and a stop condition |

## 2. Building a scene

- **Add objects** by dragging them from the *Object Library* into the viewport (they rest on the
  work plane or snap to surfaces), by double-clicking a library item, or with *Objects ▸ Add Object*.
- **Select** objects by clicking, or in the *Scene Hierarchy*. Shift-click adds and Ctrl-click
  toggles. With the Select tool (Q), dragging a rectangle on empty space selects every visible
  object whose centre is inside it (Shift adds, Ctrl toggles).
- **Transform** them with the gizmo: Q select, W move, E rotate, R scale; toggle global/local axes in
  the toolbar.
  Grid, rotation and scale snapping, and the work plane, are in *Scene ▸ Snapping*.
- **Place numerically** with *Objects ▸ Place Relative To…* (also in the viewport's context menu):
  choose another object, a reference frame or *Global*, then enter the offset along the
  reference's axes (or the global axes) and, optionally, a rotation relative to the reference. The
  dialog starts from the current placement. Several selected objects move together as one rigid
  group, the last selected one being placed.
- **Navigate** the camera:
  - left-drag on empty space orbits (in the Select tool, where it draws a selection box, use
    Alt + left-drag)
  - middle or right drag (or Shift + left drag) pans
  - the wheel zooms towards the cursor
  *View* has the standard views, *Fit Entire Scene* (Home) and camera follow modes.
- **Group** objects into assemblies (*Scene ▸ Group*, Ctrl+G). Children move with their parent,
  and a simulated assembly moves as one rigid body.
- Every edit can be undone (Ctrl+Z) and redone (Ctrl+Y). The *History* panel lists every step.

Every object has a unique, permanent ID (`OBJ-000001` …) that is never reused, even after the
object is deleted. Formulas and stop conditions refer to objects by this ID.

## 3. Properties

The **Inspector** edits the selection. Each value shows where it comes from: **U** user,
**C** calculated, **D** derived, **S** simulated. Values can be entered in any unit, for example
`5 km/h`, and the unit menu beside each field changes how the value is displayed.

- **Geometry and transform:** type parameters, and position/orientation relative to the parent or
  any reference frame.
- **Physical properties:** mass from density or entered directly, centre of mass, inertia, initial
  velocities, applied force and torque. Vectors can be expressed in any frame.
- **Collision & Joints:** the collision shape, which is separate from the visible mesh. `auto`
  uses the exact shape where one exists, otherwise the convex hull of a convex mesh (exact for
  cylinders, cones, prisms and wedges), and for a concave mesh (torus, gear, I-beam, imported
  shapes) a set of convex parts, so that a ball can pass through the hole of a ring. `convex`
  forces the single filled hull, `box` the bounding box. Approximations are marked in the inspector.
  The section also holds restitution, the friction coefficients, *Show collider* and the object's
  joints. *Friction* applies while sliding; *Static friction* (0 = the same) applies while a contact
  is at rest, so it takes more force to start an object sliding than to keep it sliding.
  *Rolling resistance* (a length, e.g. 1 mm) slows rolling objects: the resisting torque is at most
  that length times the contact force. *Spinning resistance* does the same for rotation about the
  contact normal (a spinning top or coin).
  **An object collides only if it has a collider:** rigid-body types default to `auto`, all others
  to `none`.
- **Custom properties** (*Objects ▸ Add Custom Property*): scalars, vectors, text and more, with units and
  limits. Processes can read them, and update processes can write them.
- **Display:** labels, overlays (velocity and force arrows, centre of mass, local axes) and
  trajectory markers.

## 4. Processes: what happens during a simulation

A *process* is an approved, documented calculation applied to target objects at every step.
*Processes ▸ Assign Process* lists them by category, with the formula, reads/writes and
preconditions of each:
- **Kinematics:** constant velocity, constant acceleration, prescribed circular motion,
  oscillation and rotations.
- **Dynamics:** uniform gravity, central gravity, applied force, viscous damping, two-body spring,
  anchored spring, and a uniform force field in a region.
- **Formula library:** quadratic and wind drag, N-body gravity, periodic force, tether, Lissajous
  motion, discrete update rules and an energy monitor.

Targets are explicit objects, or all objects of a type or with a tag. Parameters can be edited
while a run is going.

*Processes ▸ Validate Processes* and the *Errors & Warnings* panel report problems before a run:
- two processes writing the same quantity
- missing objects
- dependency cycles
- unit errors in formulas
- unsuitable targets

*Processes ▸ Dependencies & Diagnostics* shows what each process reads and writes, and in which order
update processes run. *Processes ▸ Global Parameters* defines project-wide constants (with units)
that formulas use as `global.<name>`.

New formula processes are written by developers only. *Tools ▸ Developer Mode* asks for the
developer passphrase, and ordinary users cannot create or change processes.

## 5. Running a simulation

The simulation toolbar holds **Δt** (simulated time per step), **Rate** (steps per real second)
and **Duration** (simulated seconds). They are independent: simulated time advances at
Rate × Δt seconds per real second when the computer keeps up. The status line shows the
achieved/requested rate. It warns *behind schedule* when the requested rate cannot be met, and
Δt is never changed to compensate.

| Command | Key | |
|---|---|---|
| Start / Resume | F5 | |
| Pause | F6 | |
| Stop | Shift+F5 | keeps the final state; the whole run becomes one undo step |
| Reset | Ctrl+F5 | back to the initial state of the run |
| Single Step | F10 | one Δt |
| Run as Fast as Possible | | steps back to back (no real-time pacing) |

*Simulation ▸ Simulation Settings* also covers:
- the integrator: semi-implicit Euler, explicit Euler, midpoint, velocity Verlet, RK4, or
  adaptive Dormand–Prince
- the gyroscopic term
- the collision and joint solver
- data recording
- breakpoints (times at which the run pauses)
- the render rate, which never affects the results

*Simulation ▸ Stop Conditions* adds formulas that pause or end the run when they *become* true,
for example:

```
object("OBJ-000002").position.z < 0[m]                       ball reaches the ground
norm(object("OBJ-000003").velocity) > 20[m/s] && t > 1[s]    too fast after the first second
```

*Check* verifies units and object references. The event appears in the console and in the
recording's event log.

*Simulation ▸ Save Simulation State* writes the current state (`.psimstate`) of a paused or
running simulation. *Restore Simulation State* continues from it later, even in another session
of the same project, and the continuation is identical to the uninterrupted run.

Most edits are applied to a running simulation at the next step: property values, process
parameters, Δt, the rate and the duration.

## 6. Rigid bodies, collisions and joints

- Gravity plus a collider are enough for objects to fall, bounce (restitution), slide and stick
  (Coulomb friction), and stack. Fast objects do not pass through thin ones: continuous collision
  detection catches pairs that would cross within one step (*Simulation Settings ▸ Collisions and
  joints*).
- **Physics ▸ Joints** connects two bodies, or a body and the world:
  - **fixed**
  - **ball**
  - **revolute** (hinge): optional angle limits and a motor
  - **prismatic** (slider): optional travel limits and a motor
  - **distance**: a rod, or a rope
  Revolute, prismatic and distance joints can have a **spring** (stiffness, damping and rest
  angle or travel; a distance spring rests at the joint's length and replaces the rigid rod).
  Any joint can **break**: give it a break force and/or torque, and it comes apart in the step in
  which it transmits more (logged as a *joint* event).
  Anchors and axes are entered in world coordinates. Limits and motors act on body A relative to B.
- A body moved only by force processes is pushed by contacts and joints. A body with prescribed
  motion pushes others but is not pushed. Objects without processes stay fixed.
- *Physics ▸ Show All Colliders / Joints / Contacts* draws the collision shapes (green exact,
  yellow approximate), joint anchors and contact forces.

For long joint simulations, prefer velocity Verlet or RK4. Semi-implicit Euler slowly loses
energy in swinging systems.

## 7. Coordinates, frames and measurements

- *Coordinates ▸ Reference Frames* defines named frames, either fixed or attached to objects. Use
  them to read and edit poses, to express vectors, and in formulas as `frame("FRM-0001")`.
- The *Relative Motion* panel shows position, velocity and acceleration of a point relative to any
  frame. It separates the Euler, centripetal and Coriolis terms.
- **Measurements** (*Tools ▸ Measure Distance / Measure Angle*, or the *Measurements* panel):
  - select two objects for a distance, or three for an angle (vertex second)
  - choose object origins or centres of mass
  - or measure between points on surfaces: *Tools ▸ Measure Distance Between Points / Measure Angle
    Between Points* (or *Points on surfaces* in the panel), then click the points on the objects in
    the viewport (for an angle: A, the vertex, B). A click near a corner or an edge of the face
    snaps onto it (hold Ctrl to place the point freely). Dragging still orbits the camera, and Esc
    cancels. Each point stays fixed on its object as it moves, turns or is rescaled.
  - values update live, also during a simulation, and appear in the viewport
  Measurements are saved with the project and can be renamed, hidden and undone.

## 8. Recording, plots and reports

During a run, selected values are recorded every N steps within a memory limit. When the limit is
reached the recorder thins out older data, keeps a recent window or stops, as configured.

- Trajectories (*Inspector ▸ Trajectory*) draw the recorded path in the chosen colour, globally or
  relative to a frame.
- *Plots ▸ Plot Workspace* holds 2D and 3D plots:
  - any variable against any variable, several objects per plot
  - logarithmic axes, fixed limits and synchronized time cursors
  - export as PNG or CSV
  *Plots ▸ Quick Plot of Selection* creates the common ones in one click.
- The *Recording* panel shows statistics per object. It exports CSV (display units) and JSON (SI,
  all channels), and writes a self-contained HTML report.

## 9. Batch runs (no GUI)

`psim-batch` (or `python -m psim.batch`) runs a project headless as fast as possible. It can
override settings, global parameters and object properties, and writes results:

```bash
psim-batch drop.psim --duration 5 --dt 0.001 --integrator rk4 \
    --global drag_cd=0.8 --set "Ball.linear_velocity=[3,0,4]" \
    --csv out.csv --var "Ball:position" --var "Ball:speed" \
    --report report.html --save-state end.psimstate --summary-json summary.json
```

It prints the steps, run time, stop conditions and a profile (milliseconds per step for each
engine stage).

**Parameter sweeps** run the project once per value, or per combination when `--sweep` is
repeated, and collect one row per run:

```bash
psim-batch drop.psim --sweep drag_cd=0.2:1.0:5 --sweep "Ball.restitution=0.5,0.9" \
    --var Ball:position.z --sweep-csv sweep.csv --jobs 4
```

- Values: `START:STOP:COUNT` (evenly spaced, both ends included), a comma list, or a JSON list
  such as `'[[1,0,0],[2,0,0]]'` for vectors. A name with a dot is an object property, otherwise
  a global parameter.
- `sweep.csv` holds the swept values, whether each run finished, its steps and end time, the stop
  conditions that fired, any error, and the final value of every `--var`.
- `--csv`, `--json`, `--report` and `--save-state` write one file per run, numbered (`out_03.csv`),
  or with `{run}` replaced in the name.
- `--jobs N` runs N simulations in parallel; the results are the same as one after another.
- A run with an invalid value is reported in its row and the others still run.

Exit status (for a sweep, the worst of its runs):
- 0: success
- 2: invalid input
- 3: the simulation failed (partial results are still written)

## 10. Files, safety and troubleshooting

- Projects are JSON files and never contain executable code; see `docs/project_format.md`.
- Autosave keeps a recovery copy while there are unsaved changes, and the next start offers to
  restore it after a crash.
- *Help ▸ System Information* shows the versions, the numerical core (native C++ or NumPy
  fallback), the OpenGL driver and the log folder. Unexpected errors are written there as
  `crash-*.txt` reports, and the session log is `psim.log`. Include them when reporting a problem.
- Slow rendering (software OpenGL, very large scenes) does not block the interface. The viewport
  draws less often so that input stays responsive. The *Performance* panel shows frame and step
  costs.
