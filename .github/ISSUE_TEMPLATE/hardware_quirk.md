---
name: Hardware quirk
about: Detect or planning output that looks wrong on your machine
title: "[quirk] <machine>: <one-line symptom>"
labels: ["hardware-quirk"]
---

## Machine

- GPU / chip:
- OS (note WSL2 if applicable):
- `frontier --version`:

## What happened

Paste the command and the surprising output (trim anything you consider
sensitive — hostnames and serials don't belong in issues either):

```text

```

## What you expected

E.g. "GB10 should produce a single unified memory node, I got two" — the
fleet runbook lists known expectations per machine.

## Profile

Attach or paste the relevant nodes/links section of the detected YAML if you
can. A failing or weird detect on new hardware is exactly the report we want.
