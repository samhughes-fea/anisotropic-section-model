<!-- Synced from the shared development standard 0.1.1. Edit it there, not here. -->
# Labels

Labels are defined in files and synced to GitHub by the `label-sync` workflow from
`main`. Labels created by hand on GitHub are deleted on the next sync.

| File | Holds | Owner |
|---|---|---|
| `.github/labels.base.yml` | Type, flag and closing labels shared by every repository | Synced from the standard |
| `.github/labels.yml` | The repository's `area:` labels and any local flags; never a label already in the base file | The repository |

## Rules

- **One type label** on every issue and pull request: `bug`, `enhancement`,
  `documentation`, `maintenance` or `dependencies`. Issue forms and Dependabot
  set it; check it when a pull request changes kind.
- **One or more area labels**, `area: <scope>`, matching the scopes in
  `.github/repo.yml` and the part of the code the change touches.
- **Flags**, whenever they apply:

| Flag | Means |
|---|---|
| `deprecation` | Retires a public name; adds the warning and the stability table row |
| `golden-change` | Moves golden numbers on purpose; the changelog says why |
| `breaking` | Removes or changes a public name (after the deprecation release) |
| `release` | The release pull request |
| `needs-decision` | Waiting on the maintainer; only the maintainer removes it |
| `good first issue` | Small and self-contained, for a new contributor |

- **Closing labels**: `duplicate`, `wontfix` (link `SCOPE.md` when out of scope),
  `question`.
- Agents apply labels when they open an issue or pull request and never remove
  `needs-decision`.

## Colours

Type labels keep GitHub's default colours; all `area:` labels share one pale
colour (`fef2c0`); review flags are red-pink (`e99695`); `needs-decision` is
yellow (`fbca04`); `release` is green (`0e8a16`).
