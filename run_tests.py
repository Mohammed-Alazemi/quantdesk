"""Run the whole test suite with one click (works in Spyder, VS Code, or `python run_tests.py`).

Unzip the project first: Python cannot import a package from inside a .zip opened by your editor.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

try:
    import pytest
except ImportError:
    sys.exit("pytest is not installed. Run   pip install -r requirements.txt   "
             "(in Spyder's console:  %pip install pytest  ) and try again.")

if __name__ == "__main__":
    sys.exit(pytest.main([os.path.join(ROOT, "tests"), "-q"]))
