"""Frozen-application entry point for the desktop GUI (see psim_studio.spec)."""
import sys

from psim.app import main

if __name__ == "__main__":
    sys.exit(main())
