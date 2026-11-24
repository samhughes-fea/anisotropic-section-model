<!-- Synced from the shared development standard 0.1.1. Edit it there, not here. -->
# Repository profile

Each repository describes itself in `.github/repo.yml`. The synced workflows and
templates read their repository-specific values from this file, so the shared
files stay identical everywhere.

```yaml
# .github/repo.yml
profile: python-package          # python-package | data-repo
package: implicit_geometry_kernel # import name; omit for data-repo
version-line: "0.9"              # releases must match this prefix; omit for none
stable: false                    # true only after the maintainer decides on 1.0.0
python: ["3.11", "3.12", "3.13"] # test matrix; first is the minimum
pypi: false                      # publish to PyPI from release.yml
docs-site: true                  # build and deploy GitHub Pages
plan-doc: docs/internal/REFACTOR_PLAN.md  # updated in every PR; omit if none
scopes:                          # Conventional Commit scopes = area labels
  - sdf
  - section
  - blade
system-packages: [libgeos-dev]   # apt packages CI installs
```

## Profiles

| Profile | For | Gets |
|---|---|---|
| `python-package` | An installable library | the common files, plus `release.yml`, `codeql.yml` and Dependabot for Actions and `requirements/` |
| `data-repo` | A data catalogue such as reference-blades | the common files, plus Dependabot for Actions |

The common files, for every profile: `pr-title.yml`, `label-sync.yml`,
`labels.base.yml`, the issue forms, `CODE_OF_CONDUCT.md`, the shared blocks of
`AGENTS.md`, `SECURITY.md` and the pull request template, and this standard in
`docs/conventions/standard/`.

## Rules

- `scopes` lists the areas a change can touch. Each scope has an `area: <scope>`
  label in the repository's local labels, and the PR title check accepts only
  these scopes plus the shared ones (`release`, `deps`, `ci`, `docs-site`,
  `examples`).
- `version-line` is the rule from [versioning](versioning.md); `release.yml`
  refuses a version outside it.
- The first sync writes `.github/repo.yml` from the standard's seed values; after
  that the repository owns it and changes it in a pull request like any other file.
- `python` and `system-packages` document the repository's local `tests.yml`;
  the synced workflows do not read them yet.
