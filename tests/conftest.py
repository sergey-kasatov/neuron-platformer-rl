import os
import pathlib
import sys

# Headless pygame: must be set before the first pygame display/surface init.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Make the package importable when pytest is launched from any directory.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
