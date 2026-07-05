# Governance

Frontier Bridge is in its pre-launch foundation phase. Governance is intentionally lightweight and will be revisited at the first public release.

## Roles

- **Maintainer:** Brian Letort ([@Brianletort](https://github.com/Brianletort)). Final decision authority on scope, schema changes, and releases during the foundation phase.
- **Contributors:** anyone submitting profiles, benchmark results, recipes, code, or documentation via pull request.

## How decisions are made

- **Schema changes** go through the RFC process. The v1 schemas (`hwprofile/v1`, `modelprofile/v1`, `plan/v1`, `benchresult/v1`) are frozen: only additive, backward-compatible changes are accepted until a v2 RFC is ratified. Breaking changes require a new RFC in `rfcs/` and a version bump.
- **Verified results** follow the results-integrity rule (see below). No exceptions, including for maintainers.
- **Everything else** (recipes, docs, adapters, profiles) is decided by maintainer review on pull requests.

## Results integrity

This single rule is the brand:

> Nothing is labeled **verified** without (a) hash-pinned model artifacts, (b) a commit-pinned runtime build, (c) a committed `benchresult/v1` file, and (d) two independent reproductions of the numbers.

Everything else is labeled **claimed** with a source, or left `null`/`unrated`.

## RFC process

1. Open a pull request adding `rfcs/NNNN-short-title.md` (copy the structure of RFC 0001).
2. Discussion happens on the PR.
3. The maintainer merges (ratified) or closes (declined) with rationale recorded in the RFC.

## Sponsorship

Frontier Bridge is sponsored by **Digital Realty** (see [NOTICE](NOTICE)). Sponsorship means funding, hardware access, and platform expertise — it does not change how technical decisions are made: the results-integrity rule, the RFC process, and the open Apache 2.0 license apply equally to sponsored and community contributions.

Public launch is gated on completing the pre-publication checklist ([docs/launch_checklist.md](docs/launch_checklist.md)), including legal, brand, and IP sign-off (Gate G0).
