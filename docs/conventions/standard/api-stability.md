<!-- Synced from the shared development standard 0.1.0. Edit it there, not here. -->
# API stability

## What is public

- Names in the `__all__` of the top-level package and of each subpackage that the
  API reference documents.
- Their call signatures, return types and documented behaviour.

Everything else is internal: modules not re-exported, names starting with `_`,
attributes reached only for tests. Internal names can change in any release.

## Changing a public name

1. Add the new name or behaviour. Adding is allowed in any release.
2. Keep the old name working for **at least one release** with a
   `DeprecationWarning` that names the replacement and the release it goes in.
   Label the pull request `deprecation` and add a row to the repository's
   "Currently deprecated" table.
3. Remove it in a later release, with `!` in the commit title, a
   `BREAKING CHANGE:` footer, a line under "Removed" in `CHANGELOG.md` and an entry
   in `MIGRATING.md`.

A bug fix that changes results is not a deprecation; it is listed in the
changelog with its bug log entry.

## Checks

- A public-API snapshot test compares the signatures of the public names with a
  golden file; a removed or changed signature fails it.
- Golden numbers catch results that move without a changelog entry.
- The docs build runs with `-W`, so a public name whose docstring does not render
  fails CI.

## Where the deprecation table lives

In the repository's own `docs/conventions/` (or D manual) page, "Currently
deprecated": name, use instead, removed in.
