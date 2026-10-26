# PyInstaller specification for PSim Studio (one-folder bundle with two executables).
#
#   python packaging/build_dist.py            # builds the native core, then runs this spec
#   pyinstaller packaging/psim_studio.spec --distpath packaging/dist --workpath packaging/build
#
# Produces dist/PSimStudio/ containing `psim-studio` (GUI, no console on Windows), `psim-batch`
# (console) and a shared `_internal/` folder; build_dist.py adds the examples and documentation.
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))   # noqa: F821 (SPECPATH is defined by PyInstaller)

hidden = collect_submodules("psim") + collect_submodules("pyqtgraph.opengl") + [
    "OpenGL.platform.glx", "OpenGL.platform.egl", "OpenGL.platform.win32", "OpenGL.arrays.numpymodule",
    "OpenGL.arrays.ctypesarrays", "OpenGL.arrays.numbers", "OpenGL.arrays.strings", "OpenGL.arrays.lists",
]
datas = collect_data_files("psim", includes=["processes/library/*.json", "processes/trusted_keys/*"])
datas += collect_data_files("pyqtgraph", includes=["**/*.png", "**/*.svg", "colors/maps/*"])
binaries = [(p, "psim/numerics") for p in
            (os.path.join(ROOT, "psim", "numerics", f) for f in os.listdir(os.path.join(ROOT, "psim", "numerics")))
            if os.path.basename(p).startswith("_psim_native")]

# Qt modules the application does not use (keeps the bundle a few hundred MB smaller)
excludes = ["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
            "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtQuick3D", "PySide6.QtMultimedia",
            "PySide6.QtPdf", "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtBluetooth",
            "PySide6.QtPositioning", "PySide6.QtLocation", "PySide6.QtSerialPort", "PySide6.QtSensors",
            "PySide6.QtTextToSpeech", "PySide6.QtDesigner", "PySide6.QtQml", "PySide6.QtQuick",
            "PySide6.QtQuickWidgets", "pytest", "pytestqt", "matplotlib", "scipy", "IPython"]


def analysis(script):
    return Analysis([os.path.join(ROOT, "packaging", script)], pathex=[ROOT], binaries=binaries,  # noqa: F821
                    datas=datas, hiddenimports=hidden, excludes=excludes, noarchive=False)


gui = analysis("entry_gui.py")
cli = analysis("entry_batch.py")
gui_pyz = PYZ(gui.pure)   # noqa: F821
cli_pyz = PYZ(cli.pure)   # noqa: F821
icon = os.path.join(ROOT, "packaging", "build", "psim.ico")    # rendered by build_dist.py
gui_exe = EXE(gui_pyz, gui.scripts, [], exclude_binaries=True, name="psim-studio", console=False,  # noqa: F821
              icon=icon if os.path.exists(icon) else None)
cli_exe = EXE(cli_pyz, cli.scripts, [], exclude_binaries=True, name="psim-batch", console=True)   # noqa: F821
COLLECT(gui_exe, cli_exe, gui.binaries, gui.datas, cli.binaries, cli.datas, name="PSimStudio")   # noqa: F821
