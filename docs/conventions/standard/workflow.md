<!-- Synced from the shared development standard 0.1.1. Edit it there, not here. -->
# Workflow: from issue to release

Every change can be traced from the release that shipped it back to the reason it
was made.

```
issue  ->  branch  ->  pull request  ->  squash commit on main  ->  release tag
 #N        any name    "Closes #N"       conventional message       vX.Y.Z
```

## 1. Issue

- Open an issue before non-trivial work: a bug, a feature, a decision, a gap. Use
  the issue forms; they set the type label. Add the area labels
  ([labels](labels.md)).
- Put it in the milestone of the release that should ship it, and it joins the
  roadmap project automatically ([planning](planning.md)).
- Work spanning several repositories is a parent issue with a sub-issue in each.
- Trivial changes (a typo, a broken link) can skip the issue.

## 2. Branch

Any name. Agents use the branch their session gives them. Branch from the latest
`main`; never pile unrelated work on one branch.

## 3. Pull request

- Title: a [Conventional Commit](commits.md); it becomes the commit on `main`.
- Body: the template ([pull-requests](pull-requests.md)), with `Closes #N` or
  `Refs #N` for each issue it answers.
- Labels and milestone match the issue.
- Required checks green, maintainer review, then squash-merge.

## 4. Records that point back

| Record | Points back by |
|---|---|
| `CHANGELOG.md` | Each line ends with the PR number, `(#81)` |
| Bug log | Each entry names its issue and fixing PR |
| Feature log | Each entry names its issue, PR and release |
| Plan doc | Each row that tracks work links its issue |
| Decision record | Links the issue or PR where it was settled |

## 5. Release

The release pull request closes the milestone. The tag and the GitHub release
follow [releasing](releasing.md); the release notes are the changelog section, so
every line links to its PR.

## Across repositories

The repositories are layered and import downwards only
([architecture](architecture.md#across-the-suite)). A change that a lower
repository must make first is done there first, released, and then the upper
repository moves its pin in its own pull request. Until a release exists, pin a
commit SHA, never a branch.
