"""PSim Studio launcher (run from a source checkout).

    python app.py [project.psim] [options]      — see `python app.py --help`

The application itself lives in psim/app.py; installed packages provide the same entry point as the
`psim-studio` command.
"""
import sys

from psim.app import main

if __name__ == "__main__":
    sys.exit(main())
