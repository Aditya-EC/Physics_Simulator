"""Application log file and handling of unexpected errors.

* ``psim.log`` (rotating, 3 × 1 MB) in the log directory receives everything the ``psim`` logger reports.
* Uncaught exceptions on the GUI thread or any worker thread are written to a timestamped
  ``crash-*.txt`` report (traceback, versions, platform, numerical backend) and logged as errors, so
  they also appear in the Console and Errors & Warnings panels. The application keeps running:
  Qt event handlers return normally, and autosave keeps the recovery file.
* ``faulthandler`` writes Python tracebacks of hard crashes (e.g. inside a graphics driver) to
  ``fault.log``.
"""
from __future__ import annotations

import datetime as _dt
import faulthandler
import logging
import logging.handlers
import os
import platform
import sys
import threading
import traceback

log = logging.getLogger("psim")
_state: dict = {}


def default_log_dir() -> str:
    """Per-user log directory (next to the autosave/recovery data)."""
    try:
        from .project.autosave import data_dir
        base = data_dir()
    except Exception:   # Qt not usable (e.g. a minimal batch environment)
        base = os.path.join(os.path.expanduser("~"), ".psim_studio")
    path = os.path.join(base, "logs")
    os.makedirs(path, exist_ok=True)
    return path


def system_info() -> dict:
    """Versions and environment, for crash reports and Help ▸ System Information."""
    from . import __version__, numerics
    info = {"application": f"PSim Studio {__version__}", "python": sys.version.split()[0],
            "platform": platform.platform(), "machine": platform.machine(),
            "numerical core": numerics.backend_description(), "frozen": bool(getattr(sys, "frozen", False))}
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:
        pass
    try:
        import PySide6
        from PySide6 import QtCore
        info["PySide6"] = PySide6.__version__
        info["Qt"] = QtCore.qVersion()
    except Exception:
        pass
    return info


def write_crash_report(exc_type, exc, tb, where: str = "main thread", log_dir: str | None = None) -> str:
    """Write a crash report and return its path."""
    log_dir = log_dir or _state.get("dir") or default_log_dir()
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = os.path.join(log_dir, f"crash-{stamp}.txt")
    lines = [f"PSim Studio crash report — {_dt.datetime.now().isoformat(timespec='seconds')}",
             f"Where: {where}", ""]
    try:
        lines += [f"{k}: {v}" for k, v in system_info().items()]
    except Exception as info_exc:   # never fail while reporting a failure
        lines.append(f"(system information unavailable: {info_exc})")
    lines += ["", "".join(traceback.format_exception(exc_type, exc, tb))]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    _state["last_report"] = path
    return path


def _handle(exc_type, exc, tb, where: str) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    try:
        path = write_crash_report(exc_type, exc, tb, where)
        log.error("Unexpected error (%s): %s: %s — details saved to %s", where, exc_type.__name__, exc, path)
    except Exception:
        sys.__excepthook__(exc_type, exc, tb)


def install(log_dir: str | None = None, *, file_logging: bool = True) -> str:
    """Install the log file, exception hooks and faulthandler. Returns the log directory. Idempotent."""
    if _state.get("installed"):
        return _state["dir"]
    log_dir = log_dir or default_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    _state["dir"] = log_dir
    if file_logging:
        handler = logging.handlers.RotatingFileHandler(os.path.join(log_dir, "psim.log"), maxBytes=1_000_000,
                                                       backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        log.addHandler(handler)
        _state["handler"] = handler
        if log.level == logging.NOTSET or log.level > logging.INFO:
            log.setLevel(logging.INFO)
    try:
        fault = open(os.path.join(log_dir, "fault.log"), "a", encoding="utf-8")   # kept open for the process
        faulthandler.enable(fault)
        _state["fault"] = fault
    except (OSError, RuntimeError):
        pass
    _state["prev_hook"], _state["prev_thread_hook"] = sys.excepthook, threading.excepthook
    sys.excepthook = lambda t, e, tb: _handle(t, e, tb, "main thread")
    threading.excepthook = lambda a: _handle(a.exc_type, a.exc_value, a.exc_traceback,
                                             f"thread {a.thread.name if a.thread else '?'}")
    _state["installed"] = True
    log.info("Session started — %s", ", ".join(f"{k} {v}" for k, v in system_info().items()))
    return log_dir


def uninstall() -> None:
    """Restore the previous hooks (used by tests)."""
    if not _state.get("installed"):
        return
    sys.excepthook = _state.pop("prev_hook")
    threading.excepthook = _state.pop("prev_thread_hook")
    handler = _state.pop("handler", None)
    if handler is not None:
        log.removeHandler(handler)
        handler.close()
    fault = _state.pop("fault", None)
    if fault is not None:
        faulthandler.disable()
        fault.close()
    _state.pop("installed", None)


def log_dir() -> str | None:
    return _state.get("dir")


def last_report() -> str | None:
    return _state.get("last_report")
