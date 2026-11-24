<!-- Synced from the shared development standard 0.1.1. Edit it there, not here. -->
# Releases and tags

A tag is a promise that consumers can pin. These rules keep every tag meaningful
and permanent; the synced `release.yml` enforces the checkable ones. In the text
below, `X.Y.Z` is a version inside the repository's version line
([versioning](versioning.md)).

## What a tag is

- **Format:** `vX.Y.Z` for a release, `vX.Y.ZrcN` for a release candidate.
  Nothing else starts with `v`.
- **Only releases are tagged.** No working tags for in-progress work. To pin
  unreleased work, a consumer pins a commit SHA on `main` and moves to the next
  release tag as soon as it exists.
- **Annotated**, with the message `<repository> X.Y.Z`.
- **On `main`.** The tagged commit is the squash commit of the release pull
  request, never a branch commit.
- **Matches the code.** The tag equals `v` + `__version__`, and `CHANGELOG.md` has
  a `## [X.Y.Z] - YYYY-MM-DD` heading ([versioning](versioning.md#changelog)).
- **Permanent.** A pushed tag is never moved, re-pointed or deleted. A mistake is
  fixed by the next patch release, and the bad release is marked in the changelog.
  A tag ruleset makes this hold on GitHub ([github-settings](github-settings.md)).
- **Made by the workflow.** Merging the release pull request is the release.
  Nobody pushes a tag by hand unless the workflow cannot (see
  [Fallback](#fallback-a-tag-by-hand)). Contributors and agents prepare the
  release pull request; whoever merges it releases.

## Between releases

On `main`, `__version__` is the next release number and the changelog's top
section, `## [Unreleased]`, collects the changes for it.

## Release steps

1. **Release pull request**, titled exactly `chore(release): prepare X.Y.Z`, with
   the `release` label. It:
   - sets `__version__` to `X.Y.Z` (it usually already is);
   - renames the changelog's top section `## [X.Y.Z] - YYYY-MM-DD`;
   - updates `MIGRATING.md` when anything was deprecated or removed;
   - updates `CITATION.cff` `version` and `date-released`, where present;
   - records the release in the plan doc;
   - closes the release milestone.
2. Required checks green, squash-merge.
3. On that push to `main`, `release.yml` finds the release commit by its subject
   and checks:
   - the subject's version equals `__version__`;
   - the version is inside the repository's version line;
   - the tag does not exist yet (if it already sits on that commit, the run stops
     and changes nothing; on another commit, the run fails);
   - the changelog has the dated heading (a release candidate needs only
     `## [X.Y.Z]`);
   - the version is below 1.0.0, unless `.github/repo.yml` has `stable: true`.
4. It builds the sdist and wheel, creates the annotated tag on the release commit
   and pushes it with `GITHUB_TOKEN`.
5. It makes the GitHub release with the changelog section as its notes and the
   sdist and wheel attached (a release candidate is a pre-release), attests their
   build provenance, and uploads to PyPI when `pypi: true` in `.github/repo.yml`
   (trusted publishing, `pypi` environment).
6. The next pull request on `main` opens a new `## [Unreleased]` section and
   bumps `__version__`.
7. Consumers move their pins to the new tag in their own pull requests.

Any other push to `main` stops after the first check; nothing is tagged.

### A release pull request merged without a tag

Run the workflow by hand (**Actions > release > Run workflow**) with the release
commit SHA. It refuses a commit that is not on `main` or not a release commit.

### A release that failed after its tag was made

Use **Re-run failed jobs**, or run the workflow by hand with the release commit
SHA. A run that finds the tag already on that commit leaves the tag alone and
completes the GitHub release and upload.

### Fallback: a tag by hand

Only when the workflow cannot run. Tag the release commit, never a later one, and
check its `__version__` first:

```bash
git tag -a vX.Y.Z -m "<repository> X.Y.Z" <release-sha>
git push origin vX.Y.Z
```

## Release order across the suite

Release from the bottom of the dependency stack up
([architecture](architecture.md#across-the-suite)): a repository is released only
after the releases it pins.

## Pinning

- Consumers pin a tag: `name @ git+https://github.com/samhughes-fea/<repo>@vX.Y.Z`,
  or a PyPI version once published.
- A git submodule points at the tagged commit, and the consumer's changelog says
  which tag.
- Never pin a branch.
