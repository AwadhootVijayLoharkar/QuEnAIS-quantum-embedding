import sys, os

# Add src/ to path once — all tests can then do `from config import ...` etc.
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))