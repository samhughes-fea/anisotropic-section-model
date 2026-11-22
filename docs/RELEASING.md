# Tags and releases

A tag is a promise that consumers can pin. These rules keep every tag meaningful
and permanent. [`release.yml`](../.github/workflows/release.yml) enforces the checkable ones. They
follow implicit-geometry-kernel's release rules, adapted for this repository.

## What a tag is

- **Format:** `vX.Y.Z` for a release, `vX.Y.ZrcN` for a release candidate.
- **Only releases are tagged.** No working tags for in-progress work. To pin
  unreleased work, a consumer pins a commit SHA on `main`.
- **Annotated**, with the message `anisotropic-section-model X.Y.Z`.
- **On `main`.** The tagged commit is the merge of the release PR.
- **Matches the code.** The tag equals `v` + `__version__` (`src/anisotropic_section_model/__init__.py`),
  and [`CHANGELOG.md`](../CHANGELOG.md) has a `## X.Y.Z - YYYY-MM-DD` heading.
  A backfilled tag (below) is the only exception, and its message says so.
- **Permanent.** A pushed tag is never moved, re-pointed or deleted. A mistake is
  fixed by the next release, and the bad one is marked in the changelog.
- **Made by the workflow.** Merging the release PR is the release:
  `release.yml` creates the tag and the GitHub release. Nobody pushes a tag by
  hand unless the workflow cannot (see [Fallback](#fallback-a-tag-by-hand)).

The rule for `main` between releases is: `__version__` is the next release
number, and the changelog's top section (`## X.Y.Z - Unreleased`) collects the
changes for it.

## Release steps

1. A release PR, titled `chore(release): prepare X.Y.Z`, sets `__version__` to
   `X.Y.Z` and dates the changelog section: `## X.Y.Z - YYYY-MM-DD`.
2. CI is green and the PR is merged into `main`. A squash merge and a merge
   commit both work: the workflow reads the PR title from the squash subject,
   or from the first body line of a `Merge pull request #N` commit.
3. On that push, `release.yml` looks at every commit the push added to `main`
   for the release PR and checks, against it:
   - the title's version equals `__version__`;
   - the version is `X.Y.Z` or `X.Y.ZrcN`;
   - the tag does not exist yet; if it already sits on that commit the run stops
     and changes nothing, and if it sits on another commit the run fails;
   - the changelog has the dated heading (a release candidate needs only the
     version heading).
   A push that carries two release PRs fails; run the workflow by hand on each,
   oldest first.
4. It builds the sdist and wheel, creates the annotated tag on the merge commit
   and pushes it with `GITHUB_TOKEN`.
5. It makes the GitHub release, with the version's changelog section as its
   notes and the sdist and wheel attached, and uploads to PyPI only when the
   repository variable `PYPI_PUBLISH` is `true`.
6. Consumers move their pins to the new tag in their own PRs.

Any other push to `main` stops after the first check; nothing is tagged.

A tag pushed with `GITHUB_TOKEN` does not start a new workflow run, so the same
run does the build and release. Runs do not share a workflow-wide concurrency
group, because GitHub keeps only one pending run per group and a later push to
`main` could then cancel a waiting release run. The tag, GitHub release and PyPI
jobs each have a group per tag.

### A release PR merged without a tag

Run the workflow by hand: **Actions > release > Run workflow**, with the merge
commit SHA. It refuses a commit that is not on `main` or is not a release PR
merge.

### A release that failed after its tag was made

Use **Re-run failed jobs** on that run, or run the workflow by hand with the
merge commit SHA. A hand run that finds the tag already on that commit leaves the
tag alone and makes or updates the GitHub release (and PyPI, skipping files
already there).

### Backfill: tagging an older release

For a release made before this workflow, run it by hand with the commit SHA and
**backfill** ticked. The run skips the release-PR title check and reads the
dated heading from the changelog on `main`, since an old release's section is
often dated after the fact. The version comes from `__version__` at that
commit. If that `__version__` was never bumped, give the tag in the **tag**
input; the tag message then says it was tagged retrospectively and names the
`__version__` the code carries. A backfilled release is never marked "latest".

### Fallback: a tag by hand

Only when the workflow cannot run. Tag the release PR's merge commit, never a
later commit: `git tag -a vX.Y.Z -m "anisotropic-section-model X.Y.Z" <merge-sha> && git push origin vX.Y.Z`.
The tag push runs `release.yml`, which checks the same rules and makes the
release.

## Repository settings (maintainer)

- **Workflow permissions.** `release.yml` asks for `contents: write` only in the
  jobs that push the tag and make the release.
- **Tag ruleset.** A tag ruleset on `v*` with **Restrict updates** and
  **Restrict deletions** makes "permanent" hold on GitHub too. If it also has
  **Restrict creations**, add **GitHub Actions** to its bypass list.
