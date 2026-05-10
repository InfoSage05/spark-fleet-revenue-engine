"""
Root conftest.py — ensures pytest can find the spark_fleet package
when running without `pip install -e .`.

This is the standard src-layout pytest workaround.
"""

import sys
from pathlib import Path

# Prepend <repo>/src so `import spark_fleet` works without an editable install.
sys.path.insert(0, str(Path(__file__).parent / "src"))
