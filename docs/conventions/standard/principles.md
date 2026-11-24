<!-- Synced from the shared development standard 0.1.1. Edit it there, not here. -->
# Principles

The tools are run like a professional software team's: every change is
**traceable**, the code stays **maintainable**, and the repository is
**understandable** to someone arriving cold, person or agent.

## Traceable

- Every change can be followed from the release that shipped it back to the reason
  it was made: release tag, changelog line, squash commit, pull request, issue.
  See [workflow](workflow.md).
- Decisions are written down where they were made, as decision records, so they
  are not reopened by accident.
- Defects are logged when found, with the fix and the test that proves it
  ([documentation](documentation.md#bug-and-feature-logs)).

## Maintainable

- One concern per pull request. Refactors leave results unchanged and prove it
  (golden numbers, snapshots).
- Public names change only after a release of deprecation warnings
  ([api-stability](api-stability.md)).
- Imports point down only, inside a repository and across the suite
  ([architecture](architecture.md)).
- Automation checks what can be checked: commit titles, labels, tags, links,
  types, docs builds.

## Understandable

- The repository can be resumed from its files alone: the plan doc, `AGENTS.md`,
  the decisions and the manual say where things stand.
- One documentation home per repository, its GitHub Pages site; no wiki.
- Plain writing: short sentences, units stated, symbols defined, no marketing words.
- Figures are committed and shown where they are explained.

## Open by default, private by rule

The tool repositories are public. Confidential inputs (for example non-public
blade data) never enter them; they are read from a private checkout that public
CI never sees ([data-and-security](data-and-security.md)).
