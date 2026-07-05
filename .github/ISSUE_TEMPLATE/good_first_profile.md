---
name: Good first profile (hardware we need)
about: Volunteer a machine for the hardware profile catalog
title: "[profile] <gpu/chip> <memory> <ram>"
labels: ["good first profile", "hardware"]
---

## Machine

- GPU / chip:
- VRAM or unified memory:
- System RAM:
- SSD (type, capacity):
- OS (note WSL2 if applicable):

## What to run

```bash
pip install -e .
frontier detect -o hardware_profiles/<your_profile_id>.yaml
frontier validate hardware_profiles/
```

Review the YAML for anything you consider sensitive, then open a PR with the
file. If `frontier detect` fails on your platform, paste the error here — a
failing detect on new hardware is exactly the bug report we want. You can also
fill in `hardware_profiles/templates/manual_template.yaml` by hand (never
guess; leave unknowns null).

## Most wanted

RTX PRO 6000 / RTX 6000 Ada, RTX 5090/4090 + large RAM, DGX Spark / GB10,
Mac Studio (M-series, 128GB+), Strix Halo, multi-GPU workstations.
