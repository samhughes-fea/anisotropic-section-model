<!-- Synced from the shared development standard 0.1.1. Edit it there, not here. -->
# Architecture

## Inside a repository

- **Down-only imports.** Subpackages form layers; a layer imports only layers below
  it. The repository's `ARCHITECTURE.md` draws its layers and a test
  (`tests/test_import_boundary.py`) enforces them.
- **Extras are not layers.** Optional dependencies (plots, CAD, Gmsh, YAML) are
  imported lazily at the call site, so a core install imports every non-plotting
  module. Core code never imports matplotlib.
- **Size.** At most three levels: package / subpackage / module. Modules stay
  under about 400 lines and subpackages under about 12 modules; split inside the
  owning folder.
- **Mirrors.** `docs/`, `tests/` and `examples/` mirror `src/<package>/` where they
  exist.
- **One way to do a thing.** A computation has one implementation; others call it.

## Across the suite

The repositories are layered too. Imports point down only:

```
implicit-geometry-kernel          geometry
  -> laminate-constitutive-model  materials and laminates
    -> shell-fem                  3D shell finite elements
      -> anisotropic-section-model  section stiffness (shell-fem in tests only)
        -> 1d-fem                 beam solver
```

- A repository never imports one above it.
- A lower repository stays generic: it never names or special-cases a consumer.
  Something a consumer has to compute on top of a lower repository is a gap in
  the lower repository, logged there as an issue.
- Plug-in seams: an upper repository defines a protocol and a lower one provides
  the default implementation, rather than hard-wiring.

## Dependencies

- Few, well-maintained runtime dependencies. NumPy everywhere; anything else is
  argued for in the pull request that adds it.
- Runtime dependencies use lower bounds only; CI requirement files pin exact
  versions and Dependabot moves them.
- Sibling repositories are pinned by release tag ([releasing](releasing.md#pinning)).
- System libraries (GEOS, gmsh, CalculiX) are documented in the U manual's install
  page and installed in CI from `system-packages` in `.github/repo.yml`.
