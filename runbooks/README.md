# Runbooks

Distributable playbooks per hardware class ([RFC 0003](../rfcs/0003-runbooks.md)).

- `*.yaml` — `runbook/v1` source of truth. Authored prose; **numbers are folded
  from committed `benchresult/v1` files, never typed in**. CI runs
  `frontier runbook verify` and rejects any number that lacks a backing result.
- `rendered/` — markdown build products from `frontier runbook render`.
  Never hand-edited, same rule as the compatibility matrix.

Find the runbook for your machine:

```bash
frontier runbook match
```

Contributing one for a hardware class we don't cover is one of the most
valuable PRs this project accepts — see [CONTRIBUTING.md](../CONTRIBUTING.md).
