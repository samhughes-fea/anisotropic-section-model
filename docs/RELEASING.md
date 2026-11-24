# Releasing

The rules are the shared standard's
[releasing](conventions/standard/releasing.md) and
[versioning](conventions/standard/versioning.md): a release is a pull request
titled `chore(release): prepare X.Y.Z`, and the synced `release.yml` tags and
releases its squash commit when it merges. Never push or move a tag by hand.

This repository's specifics:

- `__version__` lives in `src/anisotropic_section_model/__init__.py`;
  `pyproject.toml` reads it from there.
- The changelog's top section is `## [Unreleased]`; the release PR renames it
  `## [X.Y.Z] - YYYY-MM-DD`. Sections up to 0.6.1 use the older heading without
  brackets (`## 0.6.1 - 2026-11-24`), which the workflow still accepts.
- ASM depends on laminate-constitutive-model by tag, and its tests and examples
  use the geometry kernel and shell-fem by tag: release those first and bump the
  pins in `tests.yml`. 1d-fem pins ASM by tag, so it moves after a release here.
- The old workflow's backfill mode (tagging a commit made before the release
  workflow existed) is gone; every release so far (v0.2.0 to v0.6.1) is tagged.
- PyPI: `pypi: false` in `.github/repo.yml`. To publish, add a PyPI trusted
  publisher for `anisotropic-section-model` (workflow `release.yml`, environment
  `pypi`), create the `pypi` environment in the repository settings, and set
  `pypi: true`.
