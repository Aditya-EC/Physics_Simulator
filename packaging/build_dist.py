"""Build the distributable PSim Studio packages.

    python packaging/build_dist.py [--skip-native] [--run-tests] [--no-deb]

Steps:
  1. build and test the native C++ core (build_native.py), unless --skip-native
  2. optionally run the test suite (--run-tests)
  3. render the application icon (PNG + ICO) with Qt
  4. freeze the application with PyInstaller (packaging/psim_studio.spec) → packaging/dist/PSimStudio/
  5. add the examples and documentation, then smoke-test the frozen executables
  6. package:
       Linux    packaging/dist/psim-studio-<version>-linux-<arch>.tar.gz and a .deb (dpkg-deb)
       Windows  packaging/dist/psim-studio-<version>-windows-x64.zip and, when Inno Setup's ISCC.exe is on
                PATH, the installer from packaging/windows/psim_studio.iss
       macOS    packaging/dist/psim-studio-<version>-macos.zip: PSim Studio.app with the examples and docs
  Code signing (packaging/signing.py): Windows executables, native core and installer when a certificate is
  configured; the macOS app always (ad hoc without a Developer ID), notarized when notary credentials are set.

Requires PyInstaller in the build environment (`pip install pyinstaller`); it is a build tool only and not
a runtime dependency. The build machine must match the target platform (PyInstaller does not cross-compile).
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import subprocess
import sys
import glob
import tarfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DIST = os.path.join(HERE, "dist")
BUILD = os.path.join(HERE, "build")
APP_DIR = os.path.join(DIST, "PSimStudio")
sys.path.insert(0, ROOT)

from psim import APP_NAME, __version__  # noqa: E402

import signing  # noqa: E402  (packaging/signing.py)

EXE = ".exe" if os.name == "nt" else ""


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, **kw)


def render_icons() -> tuple[str, str]:
    """psim.png (256 px) and psim.ico (16–256 px) from the application's vector icon."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    from PySide6.QtCore import Qt
    app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841 (needed for QIcon rendering)
    from psim.gui import icons
    os.makedirs(BUILD, exist_ok=True)
    icon = icons.icon("cube", "#3d8bfd")
    png = os.path.join(BUILD, "psim.png")
    ico = os.path.join(BUILD, "psim.ico")
    img = QImage(256, 256, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    icon.paint(p, 0, 0, 256, 256)
    p.end()
    if not img.save(png):
        raise RuntimeError("could not write the PNG icon")
    if not img.scaled(256, 256).save(ico):    # Qt's ICO writer stores the image; Windows scales it
        print("note: Qt could not write an .ico file; the executables keep the default icon")
    if platform.system() == "Darwin" and not img.save(os.path.join(BUILD, "psim.icns")):
        print("note: Qt could not write an .icns file; the app bundle keeps the default icon")
    return png, ico


def freeze() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        sys.exit("PyInstaller is not installed in this environment: pip install pyinstaller")
    run([sys.executable, "-m", "PyInstaller", os.path.join(HERE, "psim_studio.spec"), "--noconfirm",
         "--distpath", DIST, "--workpath", os.path.join(BUILD, "pyinstaller"), "--log-level", "WARN"])


def add_documents() -> None:
    shutil.copytree(os.path.join(ROOT, "examples"), os.path.join(APP_DIR, "examples"), dirs_exist_ok=True)
    docs = os.path.join(APP_DIR, "docs")
    os.makedirs(docs, exist_ok=True)
    for f in os.listdir(os.path.join(ROOT, "docs")):
        if f.endswith((".md", ".png")):
            shutil.copy2(os.path.join(ROOT, "docs", f), docs)
    shutil.copy2(os.path.join(ROOT, "README.md"), APP_DIR)
    lic = os.path.join(ROOT, "LICENSE.txt")
    if os.path.exists(lic):
        shutil.copy2(lic, APP_DIR)


def smoke_test(exe_dir: str = APP_DIR, examples: str = os.path.join(APP_DIR, "examples")) -> None:
    """The frozen executables start, load the native core and the process library, and run a project."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    batch = os.path.join(exe_dir, "psim-batch" + EXE)
    out = os.path.join(BUILD, "smoke-summary.json")
    run([batch, os.path.join(examples, "simulation_demo.psim"), "--duration", "0.2", "--quiet",
         "--summary-json", out], env=env, cwd=BUILD)
    import json
    with open(out, encoding="utf-8") as fh:
        summary = json.load(fh)
    if not summary["finished"] or summary["steps"] <= 0:
        raise RuntimeError(f"frozen batch run did not finish: {summary}")
    gui = os.path.join(exe_dir, "psim-studio" + EXE)
    if os.name != "nt":     # a windowed .exe has no console to print --version to
        version = subprocess.check_output([gui, "--version"], env=env, text=True).strip()
        if __version__ not in version:
            raise RuntimeError(f"unexpected version output: {version!r}")
    print(f"smoke test passed ({summary['steps']} steps, native core in the bundle)")


def archive(name: str) -> str:
    os.makedirs(DIST, exist_ok=True)
    if os.name == "nt":
        path = os.path.join(DIST, name + ".zip")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for base, _dirs, files in os.walk(APP_DIR):
                for f in files:
                    full = os.path.join(base, f)
                    z.write(full, os.path.join("PSimStudio", os.path.relpath(full, APP_DIR)))
    else:
        path = os.path.join(DIST, name + ".tar.gz")
        with tarfile.open(path, "w:gz") as t:
            t.add(APP_DIR, arcname="PSimStudio")
    print("wrote", path)
    return path


def build_deb(png: str) -> str | None:
    if shutil.which("dpkg-deb") is None:
        print("dpkg-deb not found: skipping the .deb")
        return None
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine(), platform.machine())
    root = os.path.join(BUILD, "deb")
    shutil.rmtree(root, ignore_errors=True)
    opt = os.path.join(root, "opt", "psim-studio")
    shutil.copytree(APP_DIR, opt)
    bindir = os.path.join(root, "usr", "bin")
    os.makedirs(bindir)
    for exe in ("psim-studio", "psim-batch"):
        os.symlink(f"/opt/psim-studio/{exe}", os.path.join(bindir, exe))
    share = os.path.join(root, "usr", "share")
    for sub, src in ((os.path.join("applications", "psim-studio.desktop"), os.path.join(HERE, "linux",
                                                                                      "psim-studio.desktop")),
                     (os.path.join("mime", "packages", "psim-studio.xml"), os.path.join(HERE, "linux",
                                                                                      "psim-studio-mime.xml")),
                     (os.path.join("icons", "hicolor", "256x256", "apps", "psim-studio.png"), png)):
        dst = os.path.join(share, sub)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    debian = os.path.join(root, "DEBIAN")
    os.makedirs(debian)
    size_kb = sum(os.path.getsize(os.path.join(b, f)) for b, _d, fs in os.walk(opt) for f in fs) // 1024
    with open(os.path.join(debian, "control"), "w", encoding="utf-8") as fh:
        fh.write(f"Package: psim-studio\nVersion: {__version__}\nSection: science\nPriority: optional\n"
                 f"Architecture: {arch}\nInstalled-Size: {size_kb}\nMaintainer: PSim Studio <support@localhost>\n"
                 "Depends: libgl1, libegl1, libxkbcommon0, libfontconfig1, libdbus-1-3\n"
                 f"Description: {APP_NAME} — 3D physics simulation studio\n"
                 " Desktop application for building 3D scenes and simulating them with a fixed-step engine,\n"
                 " safe formula processes, rigid-body physics, data recording and plots.\n")
    refresh = ("#!/bin/sh\nset -e\nupdate-mime-database /usr/share/mime >/dev/null 2>&1 || true\n"
               "update-desktop-database >/dev/null 2>&1 || true\n")
    for script, body in (("postinst", refresh), ("postrm", refresh)):
        path = os.path.join(debian, script)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.chmod(path, 0o755)
    for base, dirs, files in os.walk(root):          # dpkg requires sane permissions
        for d in dirs:
            os.chmod(os.path.join(base, d), 0o755)
        for f in files:
            full = os.path.join(base, f)
            if os.path.islink(full) or os.path.basename(base) == "DEBIAN" and f in ("postinst", "postrm"):
                continue
            mode = os.stat(full).st_mode
            os.chmod(full, 0o755 if mode & stat.S_IXUSR else 0o644)
    out = os.path.join(DIST, f"psim-studio_{__version__}_{arch}.deb")
    run(["dpkg-deb", "--root-owner-group", "--build", root, out])
    return out


def build_windows_installer(sign=None) -> str | None:
    iscc = shutil.which("ISCC") or shutil.which("iscc")
    if iscc is None:
        print("Inno Setup (ISCC.exe) not found: install it and rerun to build the installer, or run\n"
              "  ISCC packaging\\windows\\psim_studio.iss")
        return None
    cmd = [iscc, f"/DAppVersion={__version__}"]
    if sign is not None:                      # Inno Setup signs the setup and the uninstaller it embeds
        cmd += ["/DSignInstaller", f"/Spsimsign={sign.inno_sign_tool()}"]
    signing.run_quiet(cmd + [os.path.join(HERE, "windows", "psim_studio.iss")], secret=sign and sign.password)
    return os.path.join(DIST, f"PSimStudio-{__version__}-setup.exe")


def package_windows() -> None:
    sign = signing.WindowsSigning.from_env()
    ours = [os.path.join(APP_DIR, n + EXE) for n in ("psim-studio", "psim-batch")]
    ours += glob.glob(os.path.join(APP_DIR, "_internal", "psim", "numerics", "_psim_native*.pyd"))
    if sign is not None:
        sign.sign(ours)
        signing.verify_windows(ours, sign.pfx, sign.password)
    else:
        print("note: no code-signing certificate (PSIM_WIN_CERT_PFX): the Windows packages are not signed")
    smoke_test()                              # the signed executables still run
    archive(f"psim-studio-{__version__}-windows-x64")
    setup = build_windows_installer(sign)
    if setup and sign is not None:
        signing.verify_windows([setup], sign.pfx, sign.password)


def package_macos() -> None:
    """PSim Studio.app (signed, notarized when configured) with the examples and docs, zipped with ditto."""
    cfg = signing.MacSigning.from_env()
    app = os.path.join(DIST, "PSim Studio.app")
    stage = os.path.join(BUILD, "macos-stage", "PSim Studio")
    shutil.rmtree(os.path.dirname(stage), ignore_errors=True)
    os.makedirs(stage)
    shutil.copytree(os.path.join(APP_DIR, "examples"), os.path.join(stage, "examples"))
    shutil.copytree(os.path.join(APP_DIR, "docs"), os.path.join(stage, "docs"))
    for f in ("README.md", "LICENSE.txt"):
        if os.path.exists(os.path.join(APP_DIR, f)):
            shutil.copy2(os.path.join(APP_DIR, f), stage)
    signing.sign_macos_app(app, cfg)
    signing.verify_macos_app(app, cfg)
    smoke_test(os.path.join(app, "Contents", "MacOS"), os.path.join(stage, "examples"))
    if cfg.notarize:
        submission = os.path.join(BUILD, "notarize.zip")
        run(["ditto", "-c", "-k", "--keepParent", app, submission])
        signing.notarize_macos(app, submission, cfg)
        signing.verify_macos_app(app, cfg)
    elif not cfg.ad_hoc:
        print("note: no notary credentials: the app is signed but not notarized")
    else:
        print("note: no Developer ID (PSIM_MACOS_SIGN_IDENTITY): the app is ad-hoc signed, not notarized")
    run(["ditto", app, os.path.join(stage, "PSim Studio.app")])          # ditto keeps signatures and links
    out = os.path.join(DIST, f"psim-studio-{__version__}-macos.zip")
    if os.path.exists(out):
        os.remove(out)
    run(["ditto", "-c", "-k", "--keepParent", stage, out])
    print("wrote", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skip-native", action="store_true", help="use the already built native core")
    ap.add_argument("--run-tests", action="store_true", help="run the test suite before packaging")
    ap.add_argument("--no-deb", action="store_true", help="Linux: do not build the .deb")
    args = ap.parse_args()
    if not args.skip_native:
        run([sys.executable, os.path.join(ROOT, "build_native.py")])
    if args.run_tests:
        run([sys.executable, "-m", "pytest", "-q", os.path.join(ROOT, "tests")], cwd=ROOT)
    png, _ico = render_icons()
    shutil.rmtree(APP_DIR, ignore_errors=True)
    freeze()
    add_documents()
    system = platform.system()
    if system == "Windows":
        package_windows()
    elif system == "Darwin":
        package_macos()
    else:
        smoke_test()
        archive(f"psim-studio-{__version__}-linux-{platform.machine()}")
        if not args.no_deb:
            build_deb(png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
