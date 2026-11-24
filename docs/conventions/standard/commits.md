<!-- Synced from the shared development standard 0.1.0. Edit it there, not here. -->
# Commit messages

Every commit on `main` follows [Conventional Commits 1.0](https://www.conventionalcommits.org/en/v1.0.0/).
Pull requests are squash-merged, so the **PR title** becomes the commit on `main`,
and the `pr-title` workflow checks it. Commits on a branch should follow the same
form.

## Form

```
<type>(<scope>): <summary>

<body: what changed and why, wrapped at 72 columns>

<footers>
```

- **Summary:** imperative, lower case after the colon, no full stop, under 72
  characters. Example: `fix(blade): classify a twisted flatback trailing edge as blunt`.
- **Body:** optional for small changes; say why, not how.
- **Footers:** `Closes #N` / `Refs #N`, `BREAKING CHANGE: ...`, co-author lines.

## Types

| Type | Use for | Changelog section |
|------|---------|-------------------|
| `feat` | new capability, option or public name | Added |
| `fix` | wrong results, a crash, a failed check | Fixed |
| `perf` | faster with the same results | Changed |
| `refactor` | restructuring with no behaviour change | Changed |
| `docs` | manual, docstrings, docs site, example text | Documentation |
| `test` | tests only | none |
| `build` | packaging, `pyproject.toml`, dependencies | Changed |
| `ci` | workflows | none |
| `chore` | release prep, repository housekeeping | none |
| `revert` | reverting an earlier commit | as reverted |

## Scopes

Optional. A repository's scopes are its areas, listed under `scopes` in
`.github/repo.yml` ([repo-profile](repo-profile.md)), each matching an
`area: <scope>` label. Every repository also accepts:

| Scope | For |
|---|---|
| `release` | release pull requests |
| `deps` | dependency bumps |
| `ci` | only with `build` or `chore`, for CI tooling outside workflows |
| `docs-site` | the Sphinx site and Pages build |
| `examples` | worked examples and their figures |

## Breaking changes

A removal or incompatible change of a public name adds `!` after the type or scope
(`feat(mesh)!: remove the flat mesh paths`) and a `BREAKING CHANGE:` footer
naming the replacement. Under [api-stability](api-stability.md) that only happens
after a release of deprecation warnings; the deprecation itself is a `feat` or
`refactor` with the `deprecation` label.

## Release pull requests

Titled exactly `chore(release): prepare X.Y.Z`; the release workflow looks for this
subject ([releasing](releasing.md)).

## Local check (optional)

`pre-commit install --hook-type commit-msg` runs the same check on each commit
message, where a repository's `.pre-commit-config.yaml` includes it.
