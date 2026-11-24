<!-- Synced from the shared development standard 0.1.0. Edit it there, not here. -->
# Testing

## Layout

- `tests/` mirrors `src/<package>/`: `tests/<subpackage>/test_<module>.py`.
- Shared fixtures live in `conftest.py` at the lowest level that needs them.
- `pytest -q` from the repository root runs everything that needs no private data
  and no optional system tool.

## Markers

| Marker | Means | CI |
|---|---|---|
| `slow` | over about 5 s | full run on `main`; PRs may run `-m "not slow"` |
| `reference` | needs the reference-blades checkout | skipped when absent |
| `external` | needs a system tool (gmsh, ccx) | skipped when absent |

Markers are registered in `pyproject.toml` (`--strict-markers`).

## What every change carries

- A bug fix carries a test that fails before the fix and passes after, named after
  the defect where there is a bug log ID.
- A new public name carries a test of its documented behaviour and limits.
- A numerical result carries a check against a closed form, a limiting case or a
  published value, with the tolerance stated; it is listed in the V manual
  ([documentation](documentation.md)).

## Golden numbers

Golden files (`tests/golden/*.json`) hold reference results and the public-API
snapshot.

- A refactor leaves them unchanged.
- A deliberate change regenerates them with the repository's snapshot command, in
  the same pull request, with the `golden-change` label and the reason in the
  changelog.
- Golden files never hold private inputs or results extracted from private inputs.

## Examples

Numbered examples are executable documentation. CI runs every example script on
each pull request (`examples` job). An example that needs private data or a
system tool skips with a message.

## Private data

Tests that read reference blades find the checkout through
`REFERENCE_BLADES_DIR` or a git-ignored `reference_blades/` at the repository
root, and skip cleanly when it is absent. Public CI never has it
([data-and-security](data-and-security.md)).

## Consumers

A repository lower in the stack may run its consumers' test suites against its
own checkout (a `consumers` job), so a breaking change is caught before release.
That job pins each consumer's `main`.

## Coverage

Coverage is measured on one Python version and reported in the job summary. There
is no fixed threshold; a pull request should not lower coverage of the code it
touches.
