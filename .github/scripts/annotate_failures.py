"""Turn the failures and errors of a pytest JUnit XML report into GitHub Actions error annotations.

Annotations appear on the run's summary page and through the API, so a failure can be read without the full log.
Usage: python .github/scripts/annotate_failures.py report.xml [label] [hang dump]

A hang dump (tests/conftest.py, PSIM_HANG_DUMP: the stacks of all threads of a test that ran too long) is reported
as one more annotation, without the frames of pytest itself.
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

MAX_ANNOTATIONS = 40           # GitHub shows at most 50 error annotations per step
MAX_MESSAGE = 1500


def _escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def annotate_hang(path: str, label: str) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    noise = ("/_pytest/", "/pluggy/", "/site-packages/pytest", "/dist-packages/pytest")
    kept = [ln for ln in lines if not any(n in ln for n in noise)]
    title = f"{label}: hang" if label else "hang"
    print(f"::error title={_escape(title)}::{_escape(chr(10).join(kept)[:6000])}")


def main(path: str, label: str = "", hang: str = "") -> int:
    annotate_hang(hang, label)
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        print(f"::warning::No test report to annotate ({exc})")
        return 0
    count = 0
    for case in root.iter("testcase"):
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is None:
                continue
            count += 1
            if count > MAX_ANNOTATIONS:
                continue
            name = f"{case.get('classname', '')}::{case.get('name', '')}"
            detail = (node.get("message") or "") + "\n" + (node.text or "")
            lines = detail.strip().splitlines()
            # the end of a traceback carries the assertion or exception
            body = "\n".join(lines[-25:])[-MAX_MESSAGE:]
            file = case.get("file") or case.get("classname", "").replace(".", "/") + ".py"
            title = f"{label}: {kind}" if label else kind
            print(f"::error file={file},title={_escape(title)}::{_escape(name + chr(10) + body)}")
    if count > MAX_ANNOTATIONS:
        print(f"::error::{count - MAX_ANNOTATIONS} more failures not annotated")
    print(f"{count} failing tests annotated")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
