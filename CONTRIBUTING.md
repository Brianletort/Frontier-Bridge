# Contributing to Frontier Bridge

Thanks for helping build the bridge. The most valuable early contributions, in order:

1. **Hardware profiles** from machines we don't have (RTX 6000/PRO 6000, Mac Studio, RTX 5090, Strix Halo).
2. **Benchmark results** that reproduce.
3. **Runbooks** for hardware classes we don't cover ([RFC 0003](rfcs/0003-runbooks.md)) — schema-validated, CI-checked playbooks whose numbers must trace to committed results.
4. **Recipes** — exact, working launch commands for a model/quant/runtime/hardware combination.
5. Code and docs.

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

The harness is shipped: run the loop in the [benchmark playbook](docs/benchmark_playbook.md) and submit one `benchresult/v1` JSON per run to `results/community/` via PR (there is a [PR template](.github/PULL_REQUEST_TEMPLATE/benchmark_submission.md)). The compatibility matrix is a fold over those files.

The `claimed` vs `verified` rule applies to everything: **verified** requires hash-pinned artifacts, a commit-pinned runtime, and two reproductions. Maintainers will label your submission `claimed` until it reproduces.

## Submitting a runbook

Copy the shape of [runbooks/unified-128gb.yaml](runbooks/unified-128gb.yaml). The rules that make runbooks worth trusting:

- **Numbers are folded, never authored.** Every `expected` entry needs a `source` naming a committed `benchresult/v1` result — `frontier runbook verify` (run in CI) rejects anything else. Combinations without results are listed `unmeasured: true`, with no numbers.
- Menu verdicts must agree with the committed plan files they reference.
- Model menu entries come from the catalog ([RFC 0004](rfcs/0004-catalog-admission.md)).
- Prose (diagnosis, troubleshooting) is yours — that is the human knowledge a runbook exists to carry.

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
