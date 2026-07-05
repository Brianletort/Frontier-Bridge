# Contributing to Frontier Bridge

Thanks for helping build the bridge. The most valuable early contributions, in order:

1. **Hardware profiles** from machines we don't have (RTX 6000/PRO 6000, GB10, Mac Studio, RTX 5090, Strix Halo).
2. **Benchmark results** that reproduce.
3. **Recipes** — exact, working launch commands for a model/quant/runtime/hardware combination.
4. Code and docs.

## Developer setup

```bash
git clone https://github.com/Brianletort/Frontier-Bridge.git
cd Frontier-Bridge
pip install -e ".[dev]"
pytest
frontier validate .
```

Both `pytest` and `frontier validate .` must pass before you open a PR.

## Submitting a hardware profile

Preferred path — auto-detect:

```bash
frontier detect -o hardware_profiles/<your_profile_id>.yaml
```

Review the output before submitting: remove anything you consider sensitive (hostnames, serials). If detection does not support your platform, copy `hardware_profiles/templates/manual_template.yaml`, fill in what you can measure, and set `provenance.method: manual`.

Rules:

- **Never guess.** Unknown values stay `null` or `unknown`. A sparse honest profile is more useful than a complete invented one.
- Record how you measured bandwidth numbers (tool and version) in the `measured` blocks.
- Profile IDs follow `<system>_<vram-or-unified>_<ram>` conventions, e.g. `rtx6000_96gb_64ram`.

## Submitting benchmark results

Benchmark submission lands with the Phase 4 harness. Until then, results are collected as issues. When the harness ships, submissions will be one `benchresult/v1` JSON per run, and the leaderboard is a fold over those files.

The `claimed` vs `verified` rule applies to everything: **verified** requires hash-pinned artifacts, a commit-pinned runtime, and two reproductions. Maintainers will label your submission `claimed` until it reproduces.

## Code contributions

- Python ≥ 3.10, type hints on public functions.
- Small, reviewable PRs over large rewrites.
- New or changed behavior needs a pytest test.
- Schema changes require an RFC (see [GOVERNANCE.md](GOVERNANCE.md)); v1 schemas accept additive changes only.

## Developer Certificate of Origin (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/) instead of a CLA. Sign off each commit:

```bash
git commit -s -m "Add Strix Halo hardware profile"
```

By signing off, you certify you have the right to submit the work under the Apache 2.0 license.
