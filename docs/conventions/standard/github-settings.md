<!-- Synced from the shared development standard 0.1.1. Edit it there, not here. -->
# GitHub settings

Settings that files cannot carry. The maintainer sets them once per repository;
the guidelines repository's audit workflow checks them and opens an issue on
drift.

## General

| Setting | Value |
|---|---|
| Features | Issues on; Projects on; Wiki **off**; Discussions off unless needed |
| Pull requests | Squash merging only; default commit message **Pull request title and description** |
| | Always suggest updating pull request branches: on |
| | Allow auto-merge: on |
| | Automatically delete head branches: on |
| Default branch | `main` |

## Rulesets

**`main`** (branch ruleset, target the default branch):

- Restrict deletions; block force pushes; require linear history.
- Require a pull request before merging; dismiss stale approvals on new commits;
  require conversation resolution.
- Require status checks to pass, with "require branches to be up to date"; the
  checks are listed in [ci](ci.md#required-checks).
- Bypass: the maintainer, for emergencies only.

**`v*` tags** (tag ruleset):

- Restrict updates and restrict deletions, so tags are permanent.
- If "restrict creations" is also set, GitHub Actions must be on the bypass list or
  `release.yml` cannot create tags.

Rulesets on **private** repositories need a paid plan (GitHub Pro for a personal
account). Without one, a private repository relies on the same rules by
convention and the audit report.

## Actions

| Setting | Value |
|---|---|
| Workflow permissions | Read repository contents (read-only `GITHUB_TOKEN`) |
| Allow Actions to create and approve pull requests | Off |
| Fork pull request workflows | Require approval for first-time contributors |

## Environments

`pypi` (repositories with `pypi: true`): deployment from `main` and `v*` tags
only; the PyPI project trusts this repository's `release.yml`.

## Pages

Source: GitHub Actions (`docs.yml`). The site URL is in the repository's About box
and in `README.md`.

## Code security

| Feature | Setting |
|---|---|
| Dependency graph, Dependabot alerts, Dependabot security updates | On |
| Code scanning | CodeQL via `codeql.yml` |
| Secret scanning, push protection | On |
| Private vulnerability reporting | On |

Code scanning and secret scanning are free on public repositories only.

## About box

Description in one line, the Pages URL, and topics (for example `python`,
`finite-elements`, `wind-energy`).
