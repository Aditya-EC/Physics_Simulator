"""Log file, crash reports and system information."""
from __future__ import annotations

import logging
import os
import sys
import threading

from psim import crash


def test_install_writes_log_and_crash_reports(tmp_path):
    crash.uninstall()
    try:
        d = crash.install(str(tmp_path))
        assert crash.install() == d                       # idempotent
        logging.getLogger("psim").warning("hello log")
        try:
            raise ValueError("boom")
        except ValueError:
            sys.excepthook(*sys.exc_info())
        report = crash.last_report()
        text = open(report, encoding="utf-8").read()
        assert "ValueError: boom" in text and "numerical core" in text and "Where: main thread" in text

        def worker():
            raise RuntimeError("worker failed")
        t = threading.Thread(target=worker, name="psim-test-worker")
        t.start()
        t.join()
        text = open(crash.last_report(), encoding="utf-8").read()
        assert "RuntimeError: worker failed" in text and "thread psim-test-worker" in text
        for h in logging.getLogger("psim").handlers:
            h.flush()
        log = open(os.path.join(d, "psim.log"), encoding="utf-8").read()
        assert "hello log" in log and "Unexpected error (main thread): ValueError: boom" in log
        assert os.path.exists(os.path.join(d, "fault.log"))
    finally:
        crash.uninstall()
    assert sys.excepthook is sys.__excepthook__ or not crash._state.get("installed")


def test_system_info_and_dialog(qapp):
    from psim.gui.dialogs.system_info_dialog import SystemInfoDialog, info_text
    info = crash.system_info()
    assert info["application"].startswith("PSim Studio") and "numpy" in info and "Qt" in info
    text = info_text({"renderer": "TestGPU", "version": "4.6"})
    assert "OpenGL: TestGPU · 4.6" in text and "log folder" in text
    dlg = SystemInfoDialog({"renderer": "X"})
    assert "PSim Studio" in dlg.text.toPlainText()
    dlg.deleteLater()
