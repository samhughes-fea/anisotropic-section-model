<!-- Synced from the shared development standard 0.1.1. Edit it there, not here. -->
# Continuous integration

CI runs the same checks as `CONTRIBUTING.md`, so a contributor can predict the
result before pushing.

## Workflows

| Workflow | When | Does | Profile | Source |
|---|---|---|---|---|
| `pr-title.yml` | PR opened or edited | Conventional Commit title with an allowed scope | all | synced |
| `label-sync.yml` | label files change on `main` | apply label files, delete strays | all | synced |
| `release.yml` | push to `main`, manual | tag and release on a release commit ([releasing](releasing.md)) | python-package | synced |
| `codeql.yml` | push, PR, weekly | code scanning | python-package | synced |
| `tests.yml` | push to `main`, every PR | lint, type check, tests on the Python matrix, build sdist and wheel, core install, examples | python-package | local |
| `docs.yml` | push to `main`, PRs | `sphinx-build -W`, link check, deploy to Pages from `main` | python-package with a docs site | local |
| `tool-drift.yml` | weekly | lint and type check with the newest ruff and mypy, not blocking | python-package | local |
| data checks | push, PR | catalogue schema and file checks | data-repo | local |

Synced workflows carry a header naming the standard's version; do not edit them
in place. They read the repository's values from `.github/repo.yml` with `yq`
(preinstalled on GitHub's Ubuntu runners). Local workflows depend on each
repository's system libraries, so they stay local, but they keep the job names
below and follow the hygiene rules.

## Required checks

`main`'s ruleset requires these checks, by job name: `conventional` (PR title),
the `pytest` matrix, `core-install` and `examples` from `tests.yml`, and the docs
`build` where there is a docs site. A repository adds its own (for example
`consumers`) in its ruleset.

## Workflow hygiene

- **Pin actions by commit SHA**, with the version in a comment:
  `uses: owner/action@<sha> # v1.2.3`. GitHub's own `actions/*` may use a major
  tag. Dependabot keeps both current.
- **Least privilege.** Each workflow starts with `permissions:` at the narrowest
  level (`contents: read`); a job asks for more only where it writes.
- **No secrets in pull request workflows** from forks; never use
  `pull_request_target` to run PR code.
- **Concurrency.** Cancel superseded runs on a PR branch
  (`concurrency: ${{ github.workflow }}-${{ github.ref }}`,
  `cancel-in-progress: true`); never on `main` release runs.
- **Timeouts.** Every job sets `timeout-minutes`.
- **Caching.** `actions/setup-python` with `cache: pip`.
- **Linters pinned.** ruff and mypy versions are pinned in `requirements/`;
  `tool-drift.yml` reports new releases instead of Dependabot bumping them,
  because a new rule set is a deliberate change.

## Dependabot

Monthly, grouped: one pull request for GitHub Actions, one for Python
requirements, labelled `dependencies`, titled `build(deps): ...`. Security
updates are on and arrive as soon as an advisory is published.

## Failures

- A red `main` is fixed before anything else merges.
- A flaky test is fixed or marked with an issue link; it is never skipped silently
  or retried in a loop to get green.
- After a failure, re-run only what failed while fixing it.
