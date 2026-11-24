<!-- Synced from the shared development standard 0.1.1. Edit it there, not here. -->
# Pull requests

## Title and body

- **Title:** a Conventional Commit ([commits](commits.md)). It becomes the squash
  commit, so it must read well in `git log`.
- **Body:** the repository's pull request template: *What and why*, `Closes #N`,
  the labels, and the checklist. Keep the checklist; tick what applies and strike
  through what does not, with a word on why.
- **Agent-made pull requests** say so in the body and link the session or thread
  that made them.

## Size and scope

- One concern per pull request. A refactor and a behaviour change are two pull
  requests, refactor first, so the refactor can be shown to leave results unchanged.
- Stacked pull requests name their base in the body and are merged in order.
- A pull request that moves golden numbers, deprecates a name or prepares a
  release carries the matching flag label ([labels](labels.md)).

## Checklist (shared part of the template)

- [ ] Title is a Conventional Commit; labels and milestone set
- [ ] Checks in `CONTRIBUTING.md` pass locally
- [ ] `CHANGELOG.md` updated, line ending with this PR number
- [ ] Plan doc status updated (when the repository has one)
- [ ] No public name removed or changed without a release of `DeprecationWarning`
- [ ] No private data and no downstream project names in `src/` or `tests/`
- [ ] Regenerated example figures committed only where the subject changed

Repositories add their own lines below these (for example a golden-number or
snapshot command).

## Review

- `CODEOWNERS` requests the maintainer on every pull request.
- Review comments are answered in the thread: fixed (with the commit), or why not.
  The author does not resolve a reviewer's thread with no reply.
- `needs-decision` blocks merging until the maintainer removes it.

## Merging

- **Squash-merge only**, with the PR title as the commit subject and the PR body as
  the commit body.
- Required checks must be green on a branch that is up to date with `main`. When
  `main` moves, update the branch and let CI run again.
- The head branch is deleted on merge.
- Auto-merge may be enabled on a pull request once it is approved; it merges when
  the checks pass.
