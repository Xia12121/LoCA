"""Make `import src.*` work from the repo root in tests and scripts."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
