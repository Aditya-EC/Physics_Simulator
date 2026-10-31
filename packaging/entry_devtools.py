"""Frozen-application entry point for the developer tools (psim-devtools; see psim_studio.spec)."""
import sys

from psim.devtools import main

if __name__ == "__main__":
    sys.exit(main())
