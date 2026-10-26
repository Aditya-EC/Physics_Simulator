"""Frozen-application entry point for headless batch runs (see psim_studio.spec)."""
import sys

from psim.batch import main

if __name__ == "__main__":
    sys.exit(main())
