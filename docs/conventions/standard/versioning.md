<!-- Synced from the shared development standard 0.1.0. Edit it there, not here. -->
# Versioning and the changelog

## Version numbers

- Versions are `MAJOR.MINOR.PATCH` (PEP 440 for Python packages), with release
  candidates as `X.Y.ZrcN`.
- `__version__` in the package's `__init__.py` is the single source;
  `pyproject.toml` reads it.
- **Pre-1.0.** While a repository's public API is still being worked out, it stays
  below 1.0.0. Going to 1.0.0 is a maintainer decision recorded in the
  repository's `docs/decisions/`, never an agent's step.
- **Version line.** A repository may fix a line in `.github/repo.yml`
  (`version-line: "0.9"`): every release then starts with that prefix and
  `release.yml` refuses anything else. Changing the line is a maintainer decision.
- Within pre-1.0: a deprecation or removal is allowed in any minor or patch
  release, but only after the deprecation rules in
  [api-stability](api-stability.md).

## Changelog

`CHANGELOG.md` (or `docs/CHANGELOG.md`) follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/):

```markdown
## [Unreleased]

### Added
- Smooth span laws with a station-density check (#103)

### Fixed
- ...
```

- The top section is `## [Unreleased]`; the release pull request renames it
  `## [X.Y.Z] - YYYY-MM-DD` (the release workflow also accepts the heading
  without brackets).
- Sections: Added, Changed, Deprecated, Removed, Fixed, Documentation, in that
  order, from the commit types ([commits](commits.md#types)).
- One line per change, ending with the PR number.
- A change that moves results on purpose says what moved and why.
- A bad release is marked with a line under its heading, never deleted.
- `MIGRATING.md` (where present) tells consumers what to change for each removal.
