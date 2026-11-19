#!/usr/bin/env python3
"""Run every example in order and report which ones succeeded.

Every example checks its results against its references (closed forms, published values,
shell models) and raises if one is out of tolerance, so this is also the examples' test.

Usage:
    python examples/run_all.py                 # every example, full run
    python examples/run_all.py 08 09           # only the examples whose number is given
    python examples/run_all.py --reuse         # redraw and re-check from data/results.json where
                                               # an example has it (01 and 02 always solve)
"""

import argparse
import runpy
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("only", nargs="*", help="example numbers to run (default: all)")
    parser.add_argument("--reuse", action="store_true", help="pass --reuse to the examples that support it")
    args = parser.parse_args()
    scripts = sorted(p for p in HERE.glob("[0-9][0-9]_*/*.py") if not p.name.startswith("_"))
    if args.only:
        scripts = [s for s in scripts if s.parent.name[:2] in args.only]
    failures = []
    for script in scripts:
        print(f"\n=== {script.parent.name} ===", flush=True)
        start = time.perf_counter()
        argv = sys.argv
        sys.argv = [str(script)] + (["--reuse"] if args.reuse and "--reuse" in script.read_text() else [])
        try:
            runpy.run_path(str(script), run_name="__main__")
        except (Exception, SystemExit) as exc:  # keep going so one broken example does not hide the rest
            if not (isinstance(exc, SystemExit) and exc.code in (0, None)):
                traceback.print_exc()
                failures.append(script.parent.name)
        finally:
            sys.argv = argv
        print(f"({time.perf_counter() - start:.1f} s)", flush=True)
    print(f"\n{len(scripts) - len(failures)} of {len(scripts)} examples ran and passed their checks"
          + (f"; failed: {', '.join(failures)}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
