"""Packaging metadata: entry points, package data, PyInstaller spec and desktop integration files."""
from __future__ import annotations

import configparser
import importlib
import os
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET

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
