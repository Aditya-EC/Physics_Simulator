# Acceptance tests (specification §28)

Each mandatory acceptance test has one end-to-end test in `tests/test_acceptance.py`
(`test_atNN_…`). The right-hand column lists the component suites that test the same
requirement in more depth.

Run them with:

```bash
xvfb-run -a python -m pytest tests/test_acceptance.py -v      # headless Linux
python -m pytest tests/test_acceptance.py -v                  # desktop
```

| # | Requirement | Acceptance test | Detailed tests |
|---|---|---|---|
| 1 | The application launches as a desktop application | `test_at01_application_launches_as_a_desktop_application`: main window, OpenGL renderer ready, all 14 menus | `test_gui.py::test_application_launches_with_all_menus` |
| 2 | Multiple genuine 3D object types render correctly | `test_at02_…`: cube, sphere, cylinder, cone and capsule appear at their projected positions in the framebuffer | `test_gui.py::test_objects_render_in_3d` (shading across the surface), `test_objects.py` (geometry of every type) |
| 3 | Objects can be dragged into the viewport | `test_at03_…`: library drag → drop event creates the object | `test_gui.py::test_drag_and_drop_from_library` (ghost preview, placement on the work plane) |
| 4 | Each object receives a unique persistent ID | `test_at04_…`: IDs never reused after delete, kept across save/reopen, counter continues | `test_editing.py`, `test_project_io.py` |
| 5 | Geometry, position, orientation, mass and color are editable | `test_at05_…`: edits through property paths, mass follows geometry, undo | `test_gui.py::test_inspector_edits_properties`, `test_properties.py` |
| 6 | Custom scalar and vector properties can be added | `test_at06_…`: float and vec3 custom properties, wrong vector size rejected | `test_properties.py`, `test_gui.py::test_custom_property_via_editor_appears_in_inspector` |
| 7 | A process can update velocity and position through formulas | `test_at07_…`: library formula update processes reproduce semi-implicit Euler | `test_formula_processes.py` |
| 8 | Fixed Δt and update rate are independently configurable | `test_at08_…`: both toolbar fields, simulated-time ratio = rate × Δt | `test_update_rate.py`, `test_simulation.py` |
| 9 | All applicable processes execute during every step | `test_at09_…`: three force processes each have a finite recorded contribution at every step | `test_recording.py` (process outputs sum to the total) |
| 10 | Requested and actual update rate are displayed | `test_at10_…`: toolbar status shows "achieved/requested Hz" | `test_simulation_session.py::test_gui_simulation_controls` |
| 11 | The engine reports an unachievable rate | `test_at11_…`: 100 kHz request → "behind schedule", Δt unchanged | `test_simulation.py::test_unachievable_rate_is_reported_and_dt_unchanged` |
| 12 | The clock advances by Δt for each completed step | `test_at12_…`: t_k = k·Δt exactly, final partial step ends at the duration | `test_update_rate.py`, `test_simulation.py::test_each_step_advances_clock_by_delta_time` |
| 13 | Duration is measured in simulated time | `test_at13_…`: 0.25 s simulated = 250 steps of 1 ms, independent of wall time | `test_simulation.py::test_duration_boundary_policies` |
| 14 | Rendering FPS does not set Δt or process frequency | `test_at14_…`: 5 FPS and 240 FPS give the same steps, Δt and bit-identical state | `test_simulation.py::test_results_independent_of_render_rate_and_pacing` |
| 15 | Pause, resume, stop, reset and single-step work | `test_at15_…`: all five through the session controller | `test_simulation.py::test_pause_resume_step_and_breakpoint`, `test_simulation_session.py` |
| 16 | As-fast-as-possible works independently of real-time pacing | `test_at16_…`: 2000 steps at a "1 Hz" setting finish in seconds and equal a direct engine run | `test_simulation.py` |
| 17 | Global and local coordinate systems render | `test_at17_…`: global axes and an object's local axes each change the image | `test_gui.py::test_reference_frames_render_and_menu`, `test_coordinates.py` |
| 18 | Relative coordinate transformations are numerically tested | `test_at18_…`: relative position/velocity/acceleration in an accelerating, rotating frame vs. finite differences | `test_frames.py`, `test_coordinates.py`, `test_recording_analysis.py` |
| 19 | Trajectories display using the configured color | `test_at19_…`: a magenta path appears in the framebuffer after a run | `test_trajectories.py`, `test_gui.py::test_trajectory_and_3d_plot_render_after_a_run` |
| 20 | 2D plots support configurable axes and multiple objects | `test_at20_…`: two objects' z(x), custom axis title and limits, curve data = recording | `test_plots.py` |
| 21 | 3D plots display actual trajectory data | `test_at21_…`: quick 3D trajectory plot data = recorded centre-of-mass path | `test_plots.py`, `test_gui.py` |
| 22 | Units convert correctly; incompatible formulas are reported | `test_at22_…`: km/h, lbf conversions, m→s refused, `t > 1[m]` rejected | `test_units.py`, `test_formulas.py` |
| 23 | Projects save and reopen with scene and settings intact | `test_at23_…`: full project dictionary identical after save → open (objects, processes, frames, measurements, settings, camera) | `test_project_io.py` |
| 24 | Invalid formulas and conflicting writes give useful diagnostics | `test_at24_…`: conflict names the object; syntax error shows column and caret | `test_processes.py`, `test_formulas.py`, `test_formula_processes.py` |
| 25 | Ordinary users cannot access process-authoring tools | `test_at25_…`: developer menu hidden; the registry refuses register/publish without a developer token | `test_formula_processes.py` (permissions, signing, forged tokens), `test_developer_gui.py` |
| 26 | Numerical integration methods pass regression tests | `test_at26_…`: observed orders 1 (semi-implicit Euler), 2 (midpoint), 4 (RK4) on a harmonic oscillator | `test_simulation.py`, `test_rigid_dynamics.py` |
| 27 | The GUI remains responsive while simulating | `test_at27_…`: event-loop gaps stay below 0.3 s during an unpaced collision run (software OpenGL) | `test_simulation_session.py` |
| 28 | Long simulations record without excessive memory | `test_at28_…`: 200 000 steps recorded every step stay within a 64 KB limit by decimation, covering the whole run | `test_recording.py` (decimate/window/stop policies) |
| 29 | Results are unchanged when only the render rate changes | `test_at29_…`: threaded runs at 2 and 120 FPS are bit-identical | `test_simulation.py` |
| 30 | Repeated runs are reproducible | `test_at30_…`: two runs of a stacking scene with drag (RK4) are bit-identical | `test_simulation.py::test_repeated_runs_are_bitwise_identical`, `test_sim_state.py` (restored runs continue bit for bit) |

Determinism holds on one machine and backend. The native C++ core and the NumPy fallback
agree to round-off (`test_native.py`, `test_physics.py`), not bit for bit.
