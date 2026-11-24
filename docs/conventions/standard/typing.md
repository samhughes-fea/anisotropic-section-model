<!-- Synced from the shared development standard 0.1.1. Edit it there, not here. -->
# Typing

mypy runs on the whole package in CI. Strictness rises one subpackage at a time;
a subpackage moves up a level only when it passes at that level, in the same pull
request that adds the annotations.

| Level | mypy settings |
|-------|---------------|
| 3, strict | `disallow_untyped_defs`, `disallow_incomplete_defs`, `check_untyped_defs` |
| 2, checked bodies | `check_untyped_defs`, `warn_unused_ignores` |
| 1, default | annotated functions only |

Each repository keeps its own table of which subpackage is at which level, and the
next steps, in its `docs/conventions/` or D manual. Plotting code may be excluded.

## Rules

- New modules are written at the level of their subpackage or higher; a
  subpackage never moves down a level.
- Use `numpy.typing.ArrayLike` for inputs that `np.asarray` accepts and
  `np.ndarray` (or `NDArray[np.float64]`) for returns.
- No blanket `# type: ignore`; give the code (`# type: ignore[attr-defined]`) and
  a reason.
- Public signatures are fully annotated whatever the level, because the API
  reference shows them.
