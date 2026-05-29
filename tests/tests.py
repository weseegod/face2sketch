#!/usr/bin/env python
"""
Test runner — discovers and runs all tests in the tests/ directory.

Usage:
    python tests/tests.py           # run all
    python tests/tests.py -v        # verbose output
    python tests/tests.py TestUNet  # run specific test class
"""

import sys
import unittest
from pathlib import Path

if __name__ == "__main__":
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))

    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir=str(start_dir), pattern="test_*.py")

    verbosity = 2 if "-v" in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)
