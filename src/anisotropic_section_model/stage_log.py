"""Wall-clock stage reports on one shared logger.

Every stage logs on ``blade.stage``. A collector keeps the reports in memory,
including nested spans. A printer writes one line when a stage ends. A stage
that is not entered is absent. A stage that raises reports the seconds up to
the error and the error text, then the exception propagates.

The level sits above INFO, so silencing INFO does not drop a stage report.
``seconds`` is wall time from :func:`time.perf_counter`. ``cpu_seconds`` is
process time from :func:`time.process_time`. Both are seconds.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

STAGE = 25
LOGGER_NAME = "blade.stage"
logging.addLevelName(STAGE, "STAGE")

_DEPTH_ATTR = "_blade_stage_depth"
_SOLVER = "solver"


def _depth_var() -> ContextVar[int]:
    """One nesting counter for every copy of this module.

    The counter lives on the shared logger, so a span opened in one package
    is nested inside a span opened in another.
    """
    log = logging.getLogger(LOGGER_NAME)
    slot = getattr(log, _DEPTH_ATTR, None)
    if isinstance(slot, ContextVar):
        return slot
    created: ContextVar[int] = ContextVar("blade_stage_depth", default=0)
    setattr(log, _DEPTH_ATTR, created)
    stored = getattr(log, _DEPTH_ATTR, created)
    return stored if isinstance(stored, ContextVar) else created


class StageSlot:
    """Measures recorded before a stage ends.

    A measure is a number. The only text measure is the solver name.
    """

    def __init__(self) -> None:
        self._measures: dict[str, int | float | str] = {}

    def set(self, name: str, value: int | float | str) -> None:
        if not isinstance(name, str) or not name or any(ch.isspace() for ch in name):
            raise ValueError("a measure name is a short token")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise TypeError("a measure is a number, or a solver name")
        if isinstance(value, str):
            if name != _SOLVER:
                raise TypeError("only the solver measure is text")
            if not value or any(ch.isspace() for ch in value):
                raise ValueError("a solver name is a short token")
            self._measures[name] = value
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("a measure must be finite")
            self._measures[name] = float(value)
            return
        self._measures[name] = int(value)

    def snapshot(self) -> Mapping[str, int | float | str]:
        return MappingProxyType(dict(self._measures))


def _freeze_measures(
    measures: Mapping[str, int | float | str] | None,
) -> Mapping[str, int | float | str]:
    if not measures:
        return MappingProxyType({})
    return MappingProxyType(dict(measures))


@dataclass(frozen=True)
class StageReport:
    """One finished stage."""

    package: str
    stage: str
    seconds: float
    error: str | None
    depth: int
    started: float
    cpu_seconds: float = 0.0
    measures: Mapping[str, int | float | str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "measures", _freeze_measures(self.measures))


@dataclass(frozen=True)
class RunHeader:
    """Interpreter and checkout facts. Not a timed stage."""

    python: str | None = None
    cpu_count: int | None = None
    revisions: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "revisions", MappingProxyType(dict(self.revisions)))


def report_stage(package: str, name: str, seconds: float, error: str | None,
                 started: float, depth: int | None = None, *,
                 cpu_seconds: float = 0.0,
                 measures: Mapping[str, int | float | str] | None = None) -> None:
    """Emit one stage report. ``depth`` defaults to the current nesting level."""
    level = _depth_var().get() if depth is None else depth
    frozen = _freeze_measures(measures)
    logging.getLogger(LOGGER_NAME).log(
        STAGE,
        "%s %s %.6fs%s",
        package,
        name,
        seconds,
        "" if error is None else f" error={error}",
        extra={
            "stage_package": package,
            "stage_name": name,
            "stage_seconds": float(seconds),
            "stage_error": error,
            "stage_depth": int(level),
            "stage_started": float(started),
            "stage_cpu_seconds": float(cpu_seconds),
            "stage_measures": dict(frozen),
        },
    )


@contextmanager
def stage(package: str, name: str) -> Iterator[StageSlot]:
    """Time ``name`` and report it when the block ends.

    The block may set a measure on the yielded slot. A caller that does not
    bind the slot is unchanged.
    """
    depth_var = _depth_var()
    depth = depth_var.get()
    token = depth_var.set(depth + 1)
    started = time.perf_counter()
    started_cpu = time.process_time()
    slot = StageSlot()
    try:
        yield slot
    except Exception as exc:
        _emit(package, name, started, started_cpu, str(exc), depth, slot)
        raise
    else:
        _emit(package, name, started, started_cpu, None, depth, slot)
    finally:
        depth_var.reset(token)


def _emit(package: str, name: str, started: float, started_cpu: float,
          error: str | None, depth: int, slot: StageSlot) -> None:
    cpu = time.process_time() - started_cpu
    report_stage(
        package, name, time.perf_counter() - started, error, started, depth,
        cpu_seconds=cpu if cpu >= 0.0 else 0.0, measures=slot.snapshot(),
    )


def _cpu_seconds(record: logging.LogRecord) -> float:
    value = getattr(record, "stage_cpu_seconds", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value) if math.isfinite(float(value)) else 0.0


def _record_measures(record: logging.LogRecord) -> Mapping[str, int | float | str]:
    raw = getattr(record, "stage_measures", None)
    if not isinstance(raw, Mapping):
        return MappingProxyType({})
    kept: dict[str, int | float | str] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue
        if isinstance(value, str) and key != _SOLVER:
            continue
        kept[key] = value
    return MappingProxyType(kept)


class StageCollector(logging.Handler):
    """Handler that keeps every ``blade.stage`` report in one list."""

    def __init__(self) -> None:
        super().__init__(level=STAGE)
        self.reports: list[StageReport] = []

    def emit(self, record: logging.LogRecord) -> None:
        name = getattr(record, "stage_name", None)
        seconds = getattr(record, "stage_seconds", None)
        if not isinstance(name, str):
            return
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            return
        error = getattr(record, "stage_error", None)
        depth = getattr(record, "stage_depth", 0)
        started = getattr(record, "stage_started", 0.0)
        package = getattr(record, "stage_package", "")
        self.reports.append(StageReport(
            str(package), name, float(seconds),
            None if error in (None, "") else str(error),
            int(depth) if isinstance(depth, int) and not isinstance(depth, bool) else 0,
            float(started) if isinstance(started, (int, float)) and not isinstance(started, bool)
            else 0.0,
            _cpu_seconds(record),
            _record_measures(record),
        ))

    def take(self) -> list[StageReport]:
        """Return the reports gathered so far and clear the list."""
        found = self.reports
        self.reports = []
        return found


@contextmanager
def collecting() -> Iterator[StageCollector]:
    """Attach a collector to ``blade.stage`` for the duration of the block."""
    log = logging.getLogger(LOGGER_NAME)
    previous = log.level
    handler = StageCollector()
    log.setLevel(STAGE)
    log.addHandler(handler)
    try:
        yield handler
    finally:
        log.removeHandler(handler)
        log.setLevel(previous)


class StagePrinter(logging.Handler):
    """Print one line when a stage ends."""

    def __init__(self) -> None:
        super().__init__(level=STAGE)

    def emit(self, record: logging.LogRecord) -> None:
        name = getattr(record, "stage_name", None)
        seconds = getattr(record, "stage_seconds", None)
        if not isinstance(name, str):
            return
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            return
        package = getattr(record, "stage_package", "")
        error = getattr(record, "stage_error", None)
        line = f"{package} {name} {float(seconds):.6f}s"
        if error not in (None, ""):
            line += f" error={error}"
        print(line, flush=True)


@contextmanager
def echoing() -> Iterator[None]:
    """Print one line for each stage that ends until the block exits.

    A printer already attached is left in place, so a nested block does not
    print the same line twice.
    """
    log = logging.getLogger(LOGGER_NAME)
    if any(isinstance(handler, StagePrinter) for handler in log.handlers):
        yield
        return
    handler = StagePrinter()
    previous = log.level
    if log.level == logging.NOTSET or log.level > STAGE:
        log.setLevel(STAGE)
    log.addHandler(handler)
    try:
        yield
    finally:
        log.removeHandler(handler)
        log.setLevel(previous)


def fold_reports(reports: Sequence[StageReport]) -> dict[str, object]:
    """One value per stage name.

    A nested report (depth above 0, or a span contained in another span) is
    dropped, so a solve inside buckling is not also the static stage. Spans of
    the same name that do not contain each other are added. An error keeps the
    seconds and the error text. CPU time and measures stay on the collected
    reports; they are not folded into this total.
    """
    top = [item for item in reports if item.depth <= 0]
    kept = [item for item in top if not _nested_in(item, top)]
    grouped: dict[str, list[StageReport]] = {}
    for item in kept:
        grouped.setdefault(item.stage, []).append(item)
    folded: dict[str, object] = {}
    for name, items in grouped.items():
        seconds = float(sum(item.seconds for item in items))
        errors = [item.error for item in items if item.error]
        if errors:
            folded[name] = {"seconds": seconds, "error": errors[0]}
        else:
            folded[name] = seconds
    return folded


def _nested_in(item: StageReport, reports: Sequence[StageReport]) -> bool:
    """True when another report's span contains this one.

    Equal spans keep the earlier object so a package and its caller, timing the
    same call, collapse to one record.
    """
    for other in reports:
        if other is item or not _contains(other, item):
            continue
        if not _contains(item, other):
            return True
        if other.seconds > item.seconds or (other.seconds == item.seconds and id(other) < id(item)):
            return True
    return False


def _contains(outer: StageReport, inner: StageReport) -> bool:
    end_outer = outer.started + outer.seconds
    end_inner = inner.started + inner.seconds
    return inner.started >= outer.started - 1e-9 and end_inner <= end_outer + 1e-9


def _sha(text: str) -> str | None:
    value = text.strip()
    lowered = value.lower()
    if 4 <= len(lowered) <= 64 and all(c in "0123456789abcdef" for c in lowered):
        return value
    return None


def _git_dir(marker: Path, worktree: Path) -> Path | None:
    if marker.is_dir():
        return marker
    if not marker.is_file():
        return None
    text = marker.read_text(encoding="utf-8", errors="replace").strip()
    if not text.startswith("gitdir:"):
        return None
    gitdir = Path(text.split(":", 1)[1].strip())
    if not gitdir.is_absolute():
        gitdir = (worktree / gitdir).resolve()
    return gitdir if gitdir.is_dir() else None


def _read_head(git: Path | None) -> str | None:
    if git is None:
        return None
    head = git / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8", errors="replace").strip()
    found = _sha(value)
    if found:
        return found
    if not value.startswith("ref:"):
        return None
    ref = value.split(":", 1)[1].strip()
    loose = git / ref
    if loose.is_file():
        return _sha(loose.read_text(encoding="utf-8", errors="replace"))
    packed = git / "packed-refs"
    if not packed.is_file():
        return None
    for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref:
            return _sha(parts[0])
    return None


def _git_sha(start: str | Path) -> str | None:
    path = Path(start).resolve()
    if path.is_file():
        path = path.parent
    for directory in (path, *path.parents):
        marker = directory / ".git"
        if marker.exists():
            return _read_head(_git_dir(marker, directory))
    return None


def run_header(packages: Mapping[str, str | Path] | None = None) -> RunHeader:
    """Python version, CPU count, and a git SHA for each checkout that has one.

    A package whose path is not inside a git checkout is omitted. This is a
    record of the run, not a timed stage.
    """
    revisions: dict[str, str] = {}
    if packages:
        for name, path in packages.items():
            if not isinstance(name, str) or not name:
                continue
            sha = _git_sha(path)
            if sha:
                revisions[name] = sha
    return RunHeader(python=sys.version.split()[0], cpu_count=os.cpu_count(), revisions=revisions)
