#!/usr/bin/env python3
"""Run every example in order and report which ones succeeded.

Usage: python examples/run_all.py
"""

import runpy
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    scripts = sorted(p for p in HERE.glob("[0-9][0-9]_*/*.py"))
    failures = []
    for script in scripts:
        print(f"\n=== {script.parent.name} ===")
        start = time.perf_counter()
        try:
            runpy.run_path(str(script), run_name="__main__")
        except Exception:  # keep going so one broken example does not hide the rest
            traceback.print_exc()
            failures.append(script.parent.name)
        print(f"({time.perf_counter() - start:.1f} s)")
    print(f"\n{len(scripts) - len(failures)} of {len(scripts)} examples ran" + (f"; failed: {', '.join(failures)}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
