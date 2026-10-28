"""Packaging metadata: entry points, package data, PyInstaller spec and desktop integration files."""
from __future__ import annotations

import configparser
import importlib
import os
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET

import pytest

import psim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)


def test_version_and_entry_points_resolve():
    meta = _pyproject()["project"]
    assert meta["version"] == psim.__version__
    for group in ("gui-scripts", "scripts"):
        for name, target in meta[group].items():
            module, func = target.split(":")
            assert callable(getattr(importlib.import_module(module), func)), name
    deps = " ".join(meta["dependencies"])
    for req in ("PySide6", "PyOpenGL", "numpy", "pyqtgraph"):
        assert req in deps


def test_package_data_covers_runtime_files():
    data = _pyproject()["tool"]["setuptools"]["package-data"]
    lib = os.path.join(ROOT, "psim", "processes", "library")
    assert "library/*.json" in data["psim.processes"] and any(f.endswith(".json") for f in os.listdir(lib))
    assert any(p.startswith("_psim_native") for p in data["psim.numerics"])


def test_module_entry_points_run():
    for args in (["-m", "psim", "--version"], ["-m", "psim.batch", "--help"]):
        r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
    assert psim.__version__ in subprocess.run([sys.executable, "-m", "psim", "--version"], cwd=ROOT,
                                              capture_output=True, text=True, timeout=60).stdout


def test_pyinstaller_spec_and_entry_scripts_compile():
    for f in ("psim_studio.spec", "entry_gui.py", "entry_batch.py", "build_dist.py"):
        with open(os.path.join(ROOT, "packaging", f), encoding="utf-8") as fh:
            compile(fh.read(), f, "exec")


def test_desktop_mime_and_installer_files():
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp.read(os.path.join(ROOT, "packaging", "linux", "psim-studio.desktop"), encoding="utf-8")
    entry = cp["Desktop Entry"]
    assert entry["Exec"].startswith("psim-studio") and entry["MimeType"] == "application/x-psim-project;"
    tree = ET.parse(os.path.join(ROOT, "packaging", "linux", "psim-studio-mime.xml"))
    globs = {g.get("pattern") for g in tree.iter("{http://www.freedesktop.org/standards/shared-mime-info}glob")}
    assert globs == {"*.psim", "*.psimstate"}
    iss = open(os.path.join(ROOT, "packaging", "windows", "psim_studio.iss"), encoding="utf-8").read()
    assert "psim-studio.exe" in iss and ".psim" in iss and "{#AppVersion}" in iss


def _signing():
    import importlib
    sys.path.insert(0, os.path.join(ROOT, "packaging"))
    try:
        return importlib.import_module("signing")
    finally:
        sys.path.remove(os.path.join(ROOT, "packaging"))


def test_windows_signing_commands(monkeypatch):
    sg = _signing()
    monkeypatch.setattr(sg, "find_signtool", lambda: r"C:\Program Files (x86)\Windows Kits\10\bin\x64\signtool.exe")
    cfg = sg.WindowsSigning("cert.pfx", "pw")
    cmd = cfg.sign_command("signtool")
    assert cmd[:6] == ["signtool", "sign", "/fd", "SHA256", "/f", "cert.pfx"] and cmd[6:8] == ["/p", "pw"]
    assert cmd[8:] == ["/tr", sg.DEFAULT_TIMESTAMP_URL, "/td", "SHA256"]       # RFC 3161, SHA-256
    assert "/tr" not in sg.WindowsSigning("cert.pfx", "", timestamp_url="").sign_command("signtool")
    inno = cfg.inno_sign_tool()
    assert inno.startswith('"C:\\Program Files (x86)') and inno.endswith(" $f")    # quoted path, Inno's file
    monkeypatch.delenv("PSIM_WIN_CERT_PFX", raising=False)
    assert sg.WindowsSigning.from_env() is None                                # no certificate: no signing
    monkeypatch.setenv("PSIM_WIN_CERT_PFX", os.path.join(ROOT, "missing.pfx"))
    with pytest.raises(sg.SigningError):
        sg.WindowsSigning.from_env()


def test_signing_errors_never_show_the_password():
    sg = _signing()
    with pytest.raises(sg.SigningError) as err:
        sg.run_quiet([sys.executable, "-c", "import sys; print(sys.argv[1]); sys.exit(3)", "s3cret-pw"],
                     secret="s3cret-pw")
    assert "s3cret-pw" not in str(err.value) and "***" in str(err.value)


def test_macos_signing_configuration(monkeypatch):
    sg = _signing()
    for k in ("PSIM_MACOS_SIGN_IDENTITY", "PSIM_MACOS_NOTARY_APPLE_ID", "PSIM_MACOS_NOTARY_PASSWORD",
              "PSIM_MACOS_NOTARY_TEAM_ID"):
        monkeypatch.delenv(k, raising=False)
    cfg = sg.MacSigning.from_env()
    assert cfg.ad_hoc and not cfg.notarize and cfg.codesign_args() == ["codesign", "--force", "--sign", "-"]
    monkeypatch.setenv("PSIM_MACOS_SIGN_IDENTITY", "Developer ID Application: Test (ABCDE12345)")
    cfg = sg.MacSigning.from_env()
    args = cfg.codesign_args()
    assert not cfg.notarize                                                    # no notary credentials
    assert args[args.index("--options") + 1] == "runtime" and "--timestamp" in args   # needed to notarize
    assert args[args.index("--entitlements") + 1] == sg.ENTITLEMENTS and args[-1].startswith("Developer ID")
    for k, v in (("PSIM_MACOS_NOTARY_APPLE_ID", "a@b.c"), ("PSIM_MACOS_NOTARY_PASSWORD", "x"),
                 ("PSIM_MACOS_NOTARY_TEAM_ID", "ABCDE12345")):
        monkeypatch.setenv(k, v)
    assert sg.MacSigning.from_env().notarize
    monkeypatch.setenv("PSIM_MACOS_SIGN_IDENTITY", "")
    assert not sg.MacSigning.from_env().notarize                              # ad hoc apps cannot be notarized
    import plistlib
    with open(sg.ENTITLEMENTS, "rb") as fh:
        ent = plistlib.load(fh)
    assert ent == {"com.apple.security.cs.allow-unsigned-executable-memory": True,
                   "com.apple.security.cs.disable-library-validation": True}


def test_macos_bundle_is_signed_inside_out(tmp_path):
    sg = _signing()
    app = tmp_path / "PSim Studio.app"
    files = {"Contents/MacOS/psim-studio": b"\xcf\xfa\xed\xfe....",
             "Contents/Frameworks/QtCore.framework/Versions/A/QtCore": b"\xca\xfe\xba\xbe....",
             "Contents/Frameworks/psim/numerics/_psim_native.so": b"\xcf\xfa\xed\xfe....",
             "Contents/Resources/README.md": b"# text"}
    for rel, data in files.items():
        p = app / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    os.symlink(app / "Contents/MacOS/psim-studio", app / "Contents/MacOS/link")
    found = [os.path.relpath(p, app) for p in sg.macho_files(str(app))]
    assert found[-1] == os.path.join("Contents", "MacOS", "psim-studio")      # shallowest last
    assert set(found) == {os.path.normpath(k) for k in files if not k.endswith(".md")}   # no text, no links
    assert [f.count(os.sep) for f in found] == sorted((f.count(os.sep) for f in found), reverse=True)


def test_installer_signs_setup_and_uninstaller_when_asked():
    iss = open(os.path.join(ROOT, "packaging", "windows", "psim_studio.iss"), encoding="utf-8").read()
    start = iss.index("#ifdef SignInstaller")
    block = iss[start:iss.index("#endif", start)]
    assert "SignTool=psimsign" in block and "SignedUninstaller=yes" in block
    with open(os.path.join(ROOT, "packaging", "signing.py"), encoding="utf-8") as fh:
        compile(fh.read(), "signing.py", "exec")
