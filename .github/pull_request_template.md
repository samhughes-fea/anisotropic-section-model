<!-- guidelines:start (shared development standard 0.1.0; edit it there, not here) -->
<!-- Title: a Conventional Commit, e.g. "fix(scope): ...". It becomes the squash commit. -->

## What and why

Closes #

## Labels

One type, one or more areas, and any flag that applies (`deprecation`, `breaking`, `golden-change`, `release`).

## Checks

- [ ] Checks in `CONTRIBUTING.md` pass locally
- [ ] `CHANGELOG.md` updated under the next version, line ending with this PR number
- [ ] Plan doc status updated (if this repository has one)
- [ ] No public name removed or changed without a release of `DeprecationWarning` first
- [ ] No private data and no downstream project names in `src/` or `tests/`
- [ ] Regenerated example figures committed only where the example's subject changed
<!-- guidelines:end -->
