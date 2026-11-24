<!-- Synced from the shared development standard 0.1.0. Edit it there, not here. -->
# Data and security

## Private data

The tool repositories are public. Confidential inputs never enter them.

- **Never commit** non-public inputs: aerofoil coordinates, layups, planforms,
  material cards, or data extracted from them (section tables, meshes, baselines).
- **Never name** a private project, client or confidential blade anywhere in a
  public repository, including commit messages, issues and pull requests.
- **Allowed:** figures and summary results of our own models for any blade, and
  headline facts (length, rated power) of published blades.
- **Reference blades** live in the private reference-blades repository, which sets
  the data classes (`open`, `restricted`, `nda`) and the rules for consumers.
  A consumer reads them from a checkout outside version control:
  `REFERENCE_BLADES_DIR`, or a `reference_blades/` folder listed in
  `.gitignore`. `src/` never depends on it; tests and examples skip when it is
  absent.
- **Public CI never gets access** to the private data: no token, no deploy key.
  Tests that need it run locally or in private CI.
- If private data is committed by mistake: stop, tell the maintainer, and do not
  push further commits on top. Removing it from history is the maintainer's call.

## Secrets

- No credentials in code, configuration, notebooks or logs. Use repository or
  environment secrets.
- Secret scanning with push protection is on in every repository
  ([github-settings](github-settings.md)).
- Prefer OpenID Connect (PyPI trusted publishing) over stored tokens.
- A token the maintainer creates for automation is fine-grained, scoped to the
  repositories it needs, with the fewest permissions and an expiry date.

## Reporting a vulnerability

Each repository's `SECURITY.md` (synced) asks for private reports through
**Security > Report a vulnerability** and says which versions get fixes (the
latest release only). Private vulnerability reporting is enabled in settings.

## Supply chain

- Actions pinned by SHA ([ci](ci.md#workflow-hygiene)).
- Release artefacts carry build provenance attestations.
- Untrusted input files (YAML, CSV) are parsed with safe loaders; libraries never
  call `yaml.load` without `SafeLoader`.
