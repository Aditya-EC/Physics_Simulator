# PSim Studio developer guide

How to build, test, extend and package the application. The architecture overview and design
decisions are in the README (§4). This guide covers the working practices.

## Setting up

```bash
pip install -r requirements.txt     # PySide6, PyOpenGL, NumPy, pyqtgraph, pybind11, pytest, pytest-qt
python build_native.py              # CMake build of the C++ core + its unit tests, installs into psim/numerics/
python app.py                       # or: python -m psim
```

Without the native build everything still runs on the NumPy fallback (`psim/numerics/_fallback.py`),
which implements every native routine with the same algorithm. `PSIM_FORCE_PYTHON_NUMERICS=1` forces
the fallback even when the native module exists. Use it to test both backends.

## Tests

```bash
xvfb-run -a python -m pytest                          # all suites (GUI tests need an OpenGL display)
PSIM_FORCE_PYTHON_NUMERICS=1 xvfb-run -a python -m pytest   # the same on the NumPy fallback
python -m pytest tests/test_acceptance.py -v          # the 30 acceptance tests of the specification
ctest --test-dir native/build                         # C++ unit tests
```

- GUI tests use a real `MainWindow` (fixture `window` in `tests/test_gui.py`) and skip
  automatically when no OpenGL 3.3 context can be created.
- Tests run with `QStandardPaths` test mode, so they never touch the real user profile.
- Numerical tests compare against closed-form solutions or finite differences. Do not "fix" a
  failing tolerance without understanding the error.
- `docs/acceptance.md` maps every acceptance requirement to its tests.

## Code layout and conventions

- One responsibility per module. The GUI (`psim/gui`) talks to the model through `SceneEditor`
  (undoable edits), `SimulationController` (sessions) and Qt signals; it never changes engine
  state directly.
- All writes to objects go through property paths and their `PropertyDef` validation.
- The engine (`psim/simulation`) has no Qt dependency apart from the threaded runner, so it can
  be used headless (`psim.simulation.factory.build_engine`, `psim.batch`).
- Long or heavy work never runs on the GUI thread. The runner thread talks to the GUI through
  queued signals and `SimulationRunner.call` (for state save/restore).
- No `eval`/`exec` anywhere. Formulas are parsed and compiled into closures by
  `psim/processes/formula_*`.
- Match the surrounding style: type hints, short docstrings that explain *why*, and line length
  ≤ 120.

## Extending

### Object types (plugin API)

A module in `psim/plugins/` that defines `register(registry)` is imported at start-up. A failing
plugin is reported, not fatal. `psim/plugins/example_wedge.py` is the reference:

```python
def register(registry) -> None:
    registry.register(ObjectTypeDef(
        "plugin.wedge", "Wedge (Ramp)", "Basic Solids", "description …",
        [PropertyDef("length", "Length (X)", ValueType.FLOAT, 1.2, quantity="length", minimum=1e-6), …],
        mesh_generator,                      # params -> psim.scene.geometry.Mesh (outward-oriented, closed if solid)
        derived=lambda p: {"incline": ("Incline angle", value, "angle")},
    ))
```

The type then appears in the library with a rendered thumbnail and gets mass properties from its
mesh. It is saved in projects by its type ID. Plugins are trusted application code: projects can
never load plugins.

### Built-in (Python) processes

Add a `ProcessDef` in `psim/processes/built_in_processes.py`. It needs:
- `kind`: contribution / prescribed / update
- `mode`: per object / global
- typed parameters with units
- the quantities it **reads** and **writes**; the scheduler uses these for conflict detection
- a formula string for the documentation
- an evaluation function that is vectorised over the target indices

Add tests against an analytic solution in `tests/test_processes.py` or
`tests/test_force_models.py`.

### Formula processes (developer role)

Formula processes are JSON specifications (`psim/processes/library/*.json`) written in the formula
language. To author one:
1. Unlock *Tools ▸ Developer Mode* with the developer passphrase. A user role cannot register or
   publish processes; the registry enforces this.
2. Write and test the process in the process editor (*Developer ▸ New Formula Process*).
3. Publish it as an Ed25519-signed package.

The CLI is in `psim/devtools.py`. Signed packages are verified against the public keys in the user's trusted-key
folder, and in `psim/processes/trusted_keys/` when a distribution ships one (optional; the folder
is packaged automatically if present). Local installations cannot be
made tamper-proof; signing protects integrity and provenance, not secrecy.

### Native code

Native routines live in `native/src/*.cpp` and are bound in `native/src/bindings.cpp`. Every
routine needs:
- a C++ unit test (`native/tests`)
- a NumPy twin in `_fallback.py` with the same signature
- a parity test in `tests/test_native.py` or the feature's suite

## Profiling

- **In the application:** the *Performance* panel shows the render cost per frame, the step cost,
  and the mean milliseconds per step of each engine stage: updates, processes and integration,
  prescribed motion, constraints (collisions and joints), recording and stop conditions.
- **Headless:** `psim-batch project.psim --summary-json out.json` reports the same stage profile.
  Run it under `python -m cProfile -s cumtime -m psim.batch …` for function-level detail.
- Collision detection (including convex hulls), convex decomposition, contact rows, joint rows and
  the solver are C++. The remaining Python work per step is vectorised array bookkeeping: about
  0.8 ms per step for the ten-body physics demo, and 5.3 ms for 216 stacked boxes.
- A new collision shape needs the same narrowphase in `psim/physics/collision.py` (the reference)
  and `native/src/collision.cpp`; a new joint type needs its rows in `psim/physics/joint_kernel.py`
  and `native/src/joints.cpp` (`tests/test_joint_kernel.py` compares them). Hulls reach C++
  packed into flat arrays (`hull.pack_hulls`).
  `tests/test_contact_kernel.py` and `tests/test_hulls.py` compare both backends on random scenes,
  down to the order of contacts.

## Packaging and distribution

```bash
pip install pyinstaller                  # build tool only
python packaging/build_dist.py           # native build → icons → PyInstaller → smoke test → packages
```

| Platform | Output (in `packaging/dist/`) | Verified here |
|---|---|---|
| Linux | `PSimStudio/` folder, `psim-studio-<v>-linux-<arch>.tar.gz`, `psim-studio_<v>_<arch>.deb` (installs to `/opt/psim-studio`, `/usr/bin` links, desktop entry, `.psim` MIME type) | yes: built, smoke-tested, the .deb installed and removed |
| Windows | `PSimStudio/` folder, `.zip`, and `PSimStudio-<v>-setup.exe` from `packaging/windows/psim_studio.iss` (Inno Setup 6: Start-menu entries, optional desktop icon, `.psim` association, uninstaller) | by CI: built and smoke-tested; the installer has not been run on a real machine |
| macOS | `PSim Studio.app` and `psim-studio-<v>-macos.zip` (the app with the examples and docs; `psim-batch` is in `PSim Studio.app/Contents/MacOS`) | by CI: built, signed (ad hoc without a Developer ID) and smoke-tested |

- **Python wheel:** run `pip wheel . --no-deps` after `build_native.py`. It produces a
  platform-specific wheel with the native core and the entry points `psim-studio` (GUI) and
  `psim-batch`. On Debian/Ubuntu, use pip's default build isolation: the patched system setuptools
  fails with `install_layout`.
- PyInstaller does not cross-compile: build each platform on that platform. Every build runs the
  frozen `psim-batch` on an example and checks `psim-studio --version` before packaging.
### Code signing

`packaging/signing.py` signs the packages when credentials are configured; `build_dist.py` calls it. Without
them the build still succeeds: Windows packages are unsigned, the macOS app is ad-hoc signed (needed on Apple
silicon) and both platforms print a note. CI (`.github/workflows/ci.yml`, job `package`) reads the credentials
from repository secrets (*Settings → Secrets and variables → Actions*).

**Windows (Authenticode).** Buy a code-signing certificate from a certificate authority (an EV certificate
builds SmartScreen reputation at once; a standard OV certificate builds it with downloads). Export it with its
private key as a `.pfx`. Recent CA certificates come on a hardware token or cloud HSM and cannot be exported:
then sign through the CA's or Azure's signing service instead and adapt `WindowsSigning.sign_command`.

| Secret | Value |
|---|---|
| `WINDOWS_CERT_PFX_BASE64` | the `.pfx`, base64-encoded (PowerShell: `[Convert]::ToBase64String([IO.File]::ReadAllBytes("cert.pfx"))`) |
| `WINDOWS_CERT_PASSWORD` | its password |

Signed: `psim-studio.exe`, `psim-batch.exe`, the native core (`_psim_native*.pyd`), the installer and its
uninstaller (through Inno Setup's `SignTool`), with SHA-256 and an RFC 3161 timestamp
(`PSIM_WIN_TIMESTAMP_URL`, default DigiCert's), so signatures stay valid after the certificate expires.
Without a certificate, CI signs a copy of `psim-batch.exe` with a throwaway self-signed certificate
(`python packaging/signing.py selftest <exe>`) and checks the signature, so the signing path stays tested.
Check a signed file: right-click → *Properties → Digital Signatures*, or `signtool verify /pa /v <file>`.

**macOS (Developer ID and notarization).** Join the Apple Developer Program, create a *Developer ID
Application* certificate (Xcode → Settings → Accounts, or developer.apple.com), export it from Keychain Access
with its private key as a `.p12`, and create an app-specific password for your Apple ID (account.apple.com).

| Secret | Value |
|---|---|
| `MACOS_CERT_P12_BASE64` | the `.p12`, base64-encoded (`base64 -i cert.p12 \| pbcopy`) |
| `MACOS_CERT_PASSWORD` | its password |
| `MACOS_SIGN_IDENTITY` | `Developer ID Application: <name> (<team id>)` (`security find-identity -v -p codesigning`) |
| `MACOS_NOTARY_APPLE_ID`, `MACOS_NOTARY_PASSWORD`, `MACOS_NOTARY_TEAM_ID` | Apple ID, app-specific password, team id |

CI imports the certificate into a temporary keychain (deleted at the end of the job). Every Mach-O file of
`PSim Studio.app` is signed, inner files first, then the bundle, with the hardened runtime, a secure
timestamp and `packaging/macos/entitlements.plist` (executable memory for ctypes, loading the bundled
libraries). The app is then submitted with `notarytool`, and the ticket is stapled so Gatekeeper accepts it
offline. Check: `codesign --verify --deep --strict --verbose=2 "PSim Studio.app"` and
`spctl --assess --type execute --verbose "PSim Studio.app"` (accepted only when notarized).

Locally, the same environment variables as in CI apply (`PSIM_WIN_CERT_PFX`, `PSIM_WIN_CERT_PASSWORD`,
`PSIM_MACOS_SIGN_IDENTITY`, `PSIM_MACOS_NOTARY_*`; see the module's docstring).

- Release checklist:
  - bump `__version__` (`psim/__init__.py`) and `pyproject.toml` (a test checks they match)
  - run both test backends and `tests/test_acceptance.py`
  - build the packages and install one on a clean machine
