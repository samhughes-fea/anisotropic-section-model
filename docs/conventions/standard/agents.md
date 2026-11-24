<!-- Synced from the shared development standard 0.1.0. Edit it there, not here. -->
# Agents

Coding agents and people follow the same rules. Each repository has one entry
point for both.

## Files

| File | Holds |
|---|---|
| `AGENTS.md` | The rules: a shared block synced from this standard, then the repository's own hard rules and working conventions |
| `CLAUDE.md` | One line pointing to `AGENTS.md`; nothing else, so rules are never split |
| `SCOPE.md` | What belongs in the repository and what never does, with the test for deciding |
| `ARCHITECTURE.md` | Where things live; the import layers |
| `CONTRIBUTING.md` | Set-up and the exact check commands, the same ones CI runs |

`AGENTS.md` keeps the shared part between `<!-- guidelines:start -->` and
`<!-- guidelines:end -->`. The sync replaces only that block. Local rules go
below it and are numbered on from the shared ones.

## Shared rules for every agent

1. **Read before changing.** `AGENTS.md`, `SCOPE.md`, then the plan doc named in
   `.github/repo.yml`.
2. **Work through pull requests.** `main` is protected. One concern per pull
   request; a Conventional Commit title ([commits](commits.md)).
3. **Keep the plan doc current.** Update its status in the same pull request as
   the work, so anyone can resume from the repository alone.
4. **No private data.** Never commit confidential inputs; never name a private
   project ([data-and-security](data-and-security.md)).
5. **Generic source.** `src/` and `tests/` never name or special-case a downstream
   project.
6. **Deprecate before removing** ([api-stability](api-stability.md)).
7. **Golden numbers move only on purpose**, with the reason in the changelog
   ([testing](testing.md#golden-numbers)).
8. **Never remove `needs-decision`.** Only the maintainer settles a decision.
   Record a lasting one in `docs/decisions/`.
9. **Releases follow the workflow.** Agents prepare release pull requests; merging
   one is the release ([releasing](releasing.md)). Never push or move a tag.
10. **Re-run only what failed.** After a failing check, re-run that check while
    fixing it; run everything again before asking for review.

## Writing for agents

- State a rule, then its reason in one line. Agents follow reasons better than bare
  orders, and people can tell when a rule has outlived its reason.
- Name files and commands exactly. "Run the checks" is weaker than the command.
- Put a rule where an agent will be when it matters: commit rules next to the
  commit template, figure rules in the examples README.
- Keep `AGENTS.md` short (under about 120 lines). Long material goes in the manual
  and is linked.
