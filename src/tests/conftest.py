# src/tests/conftest.py
import sys
import os

# __file__ = .../src/tests/conftest.py
# dirname once  → .../src/tests/
# dirname twice → .../src/          ← this is what we want
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

print(f"[conftest] Added to path: {SRC_DIR}")  # verify during first run