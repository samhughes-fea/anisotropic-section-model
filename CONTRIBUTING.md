# Contributing

The rules are in [AGENTS.md](AGENTS.md) and the shared standard in
[docs/conventions/standard/](docs/conventions/standard/README.md): work through a
pull request with a Conventional Commit title, one concern per pull request, and a
`CHANGELOG.md` line under `## [Unreleased]` ending with the PR number.

## Checks

CI runs the same checks (`tests.yml`). System package: `libgeos-dev`. The
constitutive model, the geometry kernel and shell-fem (tests and examples only)
are installed from their tags, as pinned in `tests.yml`.

```bash
pip install -e ".[dev]"
pytest -q                               # the test suite
MPLBACKEND=Agg python examples/run_all.py 01   # one example; CI runs each
python -m build                         # sdist and wheel
```

Commit regenerated example figures only when the example's subject changed.
