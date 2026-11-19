"""Checks an example makes on its own results, so running it is also a test.

Each example compares what it computes with a reference: a closed form, a published
result or a shell model. ``Checks`` records every comparison with its tolerance and the
source of the reference, prints them as a table, and raises at the end if any is out of
tolerance, so ``run_all.py`` and CI fail when an example stops reproducing its reference.

    checks = Checks("08 isotropic sections")
    checks.close("tube GJ", section.GJ, G * 2 * pi * R**3 * t, rtol=3e-3, ref="Bredt-Batho")
    checks.true("shear centre outside the slit tube", x_sc < -R)
    checks.report()     # prints the table; AssertionError if anything failed
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Check:
    label: str
    value: float
    reference: Optional[float]
    error: Optional[float]   # relative (or absolute, when ``scale`` was given), None for a true/false check
    tolerance: Optional[float]
    source: str
    passed: bool


class Checks:
    def __init__(self, name: str):
        self.name = name
        self.items: List[Check] = []

    def close(self, label: str, value: float, reference: float, rtol: float, ref: str = "",
              scale: Optional[float] = None) -> float:
        """``|value - reference| <= rtol * (scale or |reference|)``; returns the relative error."""
        value, reference = float(value), float(reference)
        denom = abs(scale) if scale is not None else abs(reference)
        error = abs(value - reference) / denom if denom > 0 else abs(value - reference)
        passed = bool(math.isfinite(value) and error <= rtol)
        self.items.append(Check(label, value, reference, error, rtol, ref, passed))
        return error

    def true(self, label: str, condition: bool, ref: str = "", value: float = float("nan")) -> bool:
        self.items.append(Check(label, float(value), None, None, None, ref, bool(condition)))
        return bool(condition)

    @property
    def failed(self) -> List[Check]:
        return [c for c in self.items if not c.passed]

    def report(self, json_path: Optional[Path] = None) -> None:
        """Print every check; write them to ``json_path`` if given; raise if any failed."""
        width = max([len(c.label) for c in self.items] + [10])
        print(f"\nChecks, {self.name}:")
        for c in self.items:
            mark = "ok  " if c.passed else "FAIL"
            if c.error is None:
                detail = ""
            else:
                detail = f"{c.value:12.5g} vs {c.reference:12.5g}  error {c.error:8.2e} (tol {c.tolerance:.0e})"
            print(f"  {mark} {c.label:<{width}}  {detail}" + (f"  [{c.source}]" if c.source else ""))
        print(f"  {len(self.items) - len(self.failed)} of {len(self.items)} checks passed")
        if json_path is not None:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps([asdict(c) for c in self.items], indent=1))
        if self.failed:
            raise AssertionError(f"{self.name}: {len(self.failed)} check(s) failed: "
                                 + ", ".join(c.label for c in self.failed))
