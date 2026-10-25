"""Configure, build and test the native C++ core, then install it into psim/numerics/.

Usage:
    python build_native.py            # Release build + native tests
    python build_native.py --no-tests

Requires CMake >= 3.18, a C++17 compiler (MSVC 2019+ on Windows, GCC/Clang elsewhere)
and the `pybind11` Python package.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.join(ROOT, "native")
BUILD = os.path.join(NATIVE, "build")
DEST = os.path.join(ROOT, "psim", "numerics")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tests", action="store_true")
    parser.add_argument("--config", default="Release")
    args = parser.parse_args()

    run(["cmake", "-S", NATIVE, "-B", BUILD, f"-DPython_EXECUTABLE={sys.executable}",
         f"-DCMAKE_BUILD_TYPE={args.config}", f"-DPSIM_BUILD_TESTS={'OFF' if args.no_tests else 'ON'}"])
    run(["cmake", "--build", BUILD, "--config", args.config, "--parallel"])
    if not args.no_tests:
        run(["ctest", "--test-dir", BUILD, "-C", args.config, "--output-on-failure"])

    patterns = ["_psim_native*.so", "_psim_native*.pyd"]
    built = [p for pat in patterns for p in glob.glob(os.path.join(BUILD, "**", pat), recursive=True)]
    if not built:
        print("error: built module not found", file=sys.stderr)
        return 1
    for path in built:
        target = os.path.join(DEST, os.path.basename(path))
        shutil.copy2(path, target)
        print(f"installed {os.path.relpath(target, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
