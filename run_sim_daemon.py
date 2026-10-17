"""Start the Reachy Mini daemon in MuJoCo simulation mode.

Usage: reachy_test\\Scripts\\python.exe run_sim_daemon.py [extra daemon args]

The `reachy-mini-daemon.exe` launcher is blocked by the system policy, so the
daemon is started from Python instead.
"""

import sys

# The GStreamer wheel used by reachy_mini ships its own libexpat.dll, which
# shadows Anaconda's and breaks `pyexpat` if it is imported later. Loading
# pyexpat first makes sure the correct DLL is used.
import pyexpat  # noqa: F401

from reachy_mini.daemon.app.main import main

if __name__ == "__main__":
    sys.argv = ["reachy-mini-daemon", "--sim", "--scene", "minimal", *sys.argv[1:]]
    main()
