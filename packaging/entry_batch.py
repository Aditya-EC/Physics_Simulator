"""Frozen-application entry point for headless batch runs (see psim_studio.spec)."""
import multiprocessing
import sys

from psim.batch import main

if __name__ == "__main__":
    multiprocessing.freeze_support()   # sweeps with --jobs start worker processes from this executable
    sys.exit(main())
