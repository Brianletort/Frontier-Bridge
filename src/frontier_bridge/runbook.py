"""Runbooks (RFC 0003): match, render, verify.

A runbook's prose is authored; its numbers are folded from committed
benchresult/v1 files. ``verify`` is the CI gate that keeps that true: every
``expected`` entry must name a committed result whose metrics agree, and every
``model_menu`` verdict must agree with the committed plan it references.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from frontier_bridge.results import load_results
from frontier_bridge.validation import validate_instance

RUNBOOKS_DIR = "runbooks"

# Metrics compared between an expected entry and its source benchresult.
# (runbook field, extractor over the benchresult dict)
_METRIC_EXTRACTORS = {
    "decode_tps": lambda r: (r.get("metrics") or {}).get("decode_tps"),
    "ttft_ms": lambda r: (r.get("metrics") or {}).get("ttft_ms"),
    "p95_ms": lambda r: ((r.get("metrics") or {}).get("token_latency_ms") or {}).get("p95"),
    "context": lambda r: (r.get("metrics") or {}).get("context_len_tokens"),
    "usability": lambda r: r.get("usability_suggested"),
}


@dataclass
class RunbookEntry:
    path: Path
    data: dict[str, Any]

    @property
    def runbook_id(self) -> str:
        return self.data.get("runbook_id", "?")


@dataclass
class MatchReport:
    """Outcome of evaluating one runbook's matcher against one hwprofile."""

    runbook_id: str
    matched: bool
    unmet: list[str] = field(default_factory=list)


def load_runbooks(repo_root: Path) -> list[RunbookEntry]:
    directory = repo_root / RUNBOOKS_DIR
    entries: list[RunbookEntry] = []
    if not directory.is_dir():
        return entries
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema_version") == "runbook/v1":
            entries.append(RunbookEntry(path=path, data=data))
    return entries


def _within_bounds(value: Any, bounds: dict[str, Any]) -> bool:
    """Bounds test per RFC 0003: an unmeasured (null/absent) value fails a
    minimum-bound predicate — unmeasured machines match fewer runbooks."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or math.isnan(value):
        return bounds.get("min") is None
    minimum = bounds.get("min")
    maximum = bounds.get("max")
    if minimum is not None and value < minimum:
        return False
    if maximum is not None and value > maximum:
        return False
    return True


def _node_satisfies(node: dict[str, Any], predicate: dict[str, Any]) -> bool:
    if node.get("kind") != predicate.get("kind"):
        return False
    classes = predicate.get("class")
    if classes and node.get("class") not in classes:
        return False
    capacity = predicate.get("capacity_gb")
    if capacity and not _within_bounds(node.get("capacity_gb"), capacity):
        return False
    for field_name, bounds in (predicate.get("measured") or {}).items():
        measured = node.get("measured") or {}
        if not _within_bounds(measured.get(field_name), bounds):
            return False
    return True


def _describe_predicate(predicate: dict[str, Any]) -> str:
    parts = [str(predicate.get("kind"))]
    if predicate.get("class"):
        parts.append("class in " + "/".join(predicate["class"]))
    if predicate.get("capacity_gb"):
        parts.append(f"capacity_gb {predicate['capacity_gb']}")
    if predicate.get("measured"):
        parts.append(f"measured {predicate['measured']}")
    return ", ".join(parts)


def match_runbook(runbook: dict[str, Any], hwprofile: dict[str, Any]) -> MatchReport:
    """A machine matches when every require predicate is satisfied by at
    least one node in its profile."""
    nodes = hwprofile.get("nodes") or []
    unmet = []
    for predicate in (runbook.get("hardware_class") or {}).get("require", []):
        if not any(_node_satisfies(node, predicate) for node in nodes):
            unmet.append(_describe_predicate(predicate))
    return MatchReport(
        runbook_id=runbook.get("runbook_id", "?"),
        matched=not unmet,
        unmet=unmet,
    )


def verify_runbook(runbook: dict[str, Any], repo_root: Path) -> list[str]:
    """The provenance gate. Returns error strings; empty means clean."""
    errors = [f"schema: {e}" for e in validate_instance(runbook)]

    results_by_id = {
        r.get("result_id"): r for r in load_results(repo_root) if r.get("result_id")
    }

    for i, entry in enumerate(runbook.get("expected") or []):
        where = f"expected[{i}] ({entry.get('modelprofile')}/{entry.get('workload')})"
        if entry.get("unmeasured"):
            continue
        source = entry.get("source")
        result = results_by_id.get(source)
        if result is None:
            errors.append(f"{where}: source {source!r} is not a committed benchresult")
            continue
        for fieldname, extract in _METRIC_EXTRACTORS.items():
            stated = entry.get(fieldname)
            if stated is None:
                continue
            actual = extract(result)
            if stated != actual:
                errors.append(
                    f"{where}: {fieldname}={stated} disagrees with "
                    f"benchresult {source} ({actual})"
                )

    for i, item in enumerate(runbook.get("model_menu") or []):
        where = f"model_menu[{i}] ({item.get('modelprofile')})"
        plan_ref = item.get("plan_ref")
        plan_path = repo_root / plan_ref if plan_ref else None
        if plan_path is None or not plan_path.is_file():
            errors.append(f"{where}: plan_ref {plan_ref!r} does not exist")
            continue
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        if plan.get("schema_version") != "plan/v1":
            errors.append(f"{where}: plan_ref {plan_ref!r} is not a plan/v1 file")
            continue
        if plan.get("verdict") != item.get("verdict"):
            errors.append(
                f"{where}: verdict {item.get('verdict')!r} disagrees with "
                f"plan {plan_ref} ({plan.get('verdict')!r})"
            )
        plan_model = (plan.get("inputs") or {}).get("modelprofile")
        if plan_model != item.get("modelprofile"):
            errors.append(
                f"{where}: modelprofile disagrees with plan {plan_ref} ({plan_model!r})"
            )
    return errors


def _fmt(value: Any) -> str:
    return "—" if value is None else str(value)


def render_markdown(runbook: dict[str, Any]) -> str:
    """Render a runbook/v1 document to self-contained markdown.

    The YAML is the source of truth; this output is a build product and is
    never hand-edited (same rule as the compatibility matrix).
    """
    hardware_class = runbook.get("hardware_class") or {}
    lines = [
        f"# {runbook.get('title', runbook.get('runbook_id'))}",
        "",
        f"Generated by `frontier runbook render` from "
        f"`runbooks/{runbook.get('runbook_id')}.yaml` — do not edit by hand.",
        "",
        f"**Hardware class:** {hardware_class.get('name')} — "
        f"{hardware_class.get('description') or ''}".rstrip(" —"),
        "",
    ]
    known = hardware_class.get("known_profiles") or []
    if known:
        lines += ["Committed profiles known to match: " + ", ".join(f"`{p}`" for p in known), ""]

    lines += ["## Does this runbook apply to your machine?", ""]
    lines += [
        "Run `frontier runbook match` — it profiles your machine and checks it "
        "against every runbook's requirements. This class requires:",
        "",
    ]
    for predicate in hardware_class.get("require", []):
        lines.append(f"- {_describe_predicate(predicate)}")
    lines.append("")

    lines += ["## Bring-up", ""]
    for i, step in enumerate(runbook.get("diagnosis") or [], 1):
        lines.append(f"{i}. {step.get('step')}")
        if step.get("command"):
            lines += ["", "   ```bash", f"   {step['command']}", "   ```", ""]
        if step.get("check"):
            lines.append(f"   *Check:* {step['check']}")
            lines.append("")

    lines += ["## What to run", ""]
    lines += [
        "| Model | Role | Verdict | Notes |",
        "|---|---|---|---|",
    ]
    for item in runbook.get("model_menu") or []:
        lines.append(
            f"| `{item.get('modelprofile')}` | {item.get('role')} "
            f"| {item.get('verdict')} | {item.get('notes') or ''} |"
        )
    lines += [
        "",
        "Verdicts come from committed plans; refusals are listed on purpose — "
        "knowing what *not* to download is half the value.",
        "",
    ]

    expected = runbook.get("expected") or []
    if expected:
        lines += ["## Measured expectations", ""]
        lines += [
            "| Model | Workload | Decode tps | TTFT ms | p95 ms | Ctx | Usability | Source |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for entry in expected:
            if entry.get("unmeasured"):
                lines.append(
                    f"| `{entry.get('modelprofile')}` | {entry.get('workload')} "
                    f"| unmeasured | unmeasured | unmeasured | — | — | — |"
                )
            else:
                lines.append(
                    f"| `{entry.get('modelprofile')}` | {entry.get('workload')} "
                    f"| {_fmt(entry.get('decode_tps'))} | {_fmt(entry.get('ttft_ms'))} "
                    f"| {_fmt(entry.get('p95_ms'))} | {_fmt(entry.get('context'))} "
                    f"| {_fmt(entry.get('usability'))} | `{entry.get('source')}` |"
                )
        lines += [
            "",
            "Every number above is folded from a committed, hash-pinned "
            "`benchresult/v1` file (the Source column). Rows marked unmeasured "
            "have no numbers because none were measured — never estimated.",
            "",
        ]

    troubleshooting = runbook.get("troubleshooting") or []
    if troubleshooting:
        lines += ["## Troubleshooting", ""]
        for item in troubleshooting:
            lines.append(f"- **{item.get('symptom')}** — check: {item.get('check')}")
            if item.get("fix"):
                lines.append(f"  Fix: {item['fix']}")
        lines.append("")

    provenance = runbook.get("provenance") or {}
    lines += [
        "---",
        "",
        f"Authored by {provenance.get('authored_by')} · created "
        f"{provenance.get('created')}"
        + (f" · updated {provenance['updated']}" if provenance.get("updated") else ""),
        "",
    ]
    return "\n".join(lines)
