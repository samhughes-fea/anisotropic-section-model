<!-- Synced from the shared development standard 0.1.0. Edit it there, not here. -->
# Planning

GitHub Issues is the live record of work. A user-level GitHub Project ties the
repositories together; milestones tie issues to releases.

## The roadmap project

One project, **Tools roadmap**, owned by the maintainer's account, across every
tool repository.

| Field | Values |
|---|---|
| Status | Inbox, Ready, In progress, In review, Done |
| Priority | P0 (now), P1 (next), P2 (later) |
| Size | S (hours), M (days), L (a week or more) |
| Release | the milestone, mirrored from each issue |
| Repository | built in |

Views: **Board** by Status; **Roadmap** by Release; **Table** grouped by
Repository; **Decisions**, filtered to `label:needs-decision`.

Built-in project workflows:

- Auto-add issues and pull requests from every tool repository, Status **Inbox**.
- Pull request merged or issue closed: Status **Done**.
- Item reopened: Status **In progress**.
- Auto-archive items Done for 14 days.

## Milestones

- One milestone per planned release in each repository, named after the version
  (`0.9.6`), with a due date when there is one.
- Every issue planned for a release is in its milestone; the release pull request
  closes it.
- Agents may create the next milestone when they prepare a release.

## Issues

- **Issue forms** (synced): bug report, feature request, decision request, gap
  in a lower package, release plan. Blank issues are off; the chooser links to the manual and to
  private vulnerability reporting.
- **Type labels** stand in for issue types, which personal accounts do not have.
- **Sub-issues** split a large piece of work into parts, across repositories where
  needed. The parent's description says the goal and the order.
- A decision request carries `needs-decision`; once settled, the maintainer
  removes it and a decision record links the issue.
- Close an issue with a reason: completed (by a PR, `Closes #N`) or not planned
  (with a closing label).

## The plan doc

Some repositories also keep a plan doc in the repository (named in
`.github/repo.yml`), so the plan travels with the code and an agent can resume
from the repository alone. Its rows link their issues, so the document and
GitHub agree; the project board is the overview, the plan doc the detail.
