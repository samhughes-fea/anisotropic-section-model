<!-- Synced from the shared development standard 0.1.1. Edit it there, not here. -->
# Development standard

These are the conventions every `samhughes-fea` tool repository follows. They are
written once and synced into each repository's `docs/conventions/standard/`; do
not edit them there. A repository
adds what is specific to it in its own `AGENTS.md`, `SCOPE.md` and
`docs/decisions/`; where the two differ, the repository's own rule wins and should
say why.

| Standard | Covers |
|---|---|
| [principles](principles.md) | What "run like a professional team" means here |
| [repo-profile](repo-profile.md) | The per-repository values the standard reads (`.github/repo.yml`) |
| [agents](agents.md) | `AGENTS.md`, `CLAUDE.md`, how coding agents work in a repository |
| [workflow](workflow.md) | Issue, branch, pull request, squash commit, release |
| [commits](commits.md) | Conventional Commits, types, scopes, breaking changes |
| [pull-requests](pull-requests.md) | Title, body, checks, review, merging |
| [labels](labels.md) | Type, area and flag labels; label sync |
| [releasing](releasing.md) | Release pull request, tags, GitHub releases, PyPI, pinning |
| [versioning](versioning.md) | Version lines, pre-1.0, the changelog |
| [ci](ci.md) | Required checks, workflow hygiene, Dependabot, security scanning |
| [testing](testing.md) | Test layout, markers, golden numbers, examples, private data skips |
| [documentation](documentation.md) | The manual (U, R, V, D), derivations, figures, docstrings, Pages |
| [api-stability](api-stability.md) | What is public; deprecate before removing |
| [typing](typing.md) | mypy levels and the typing roadmap |
| [architecture](architecture.md) | Down-only imports, extras, module size, dependencies |
| [data-and-security](data-and-security.md) | Private data, reference blades, secrets, vulnerability reports |
| [github-settings](github-settings.md) | Rulesets, merge settings, Actions, Pages, security features |
| [planning](planning.md) | Projects, milestones, issues, sub-issues |

Words: **must** is a rule CI or review enforces; **should** is the default, with a
reason in the PR when it is not followed.
