"""Fold committed benchresult/v1 files into the compatibility matrix.

The leaderboard is a fold over result files — no separate database. Rows are
grouped by (hwprofile, modelprofile, workload); the best-status result wins the
row; verified beats claimed; more reproductions beat fewer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RESULTS_DIRS = ("results/local", "results/community", "results/verified")


@dataclass
class MatrixRow:
    hwprofile: str
    modelprofile: str
    workload: str
    status: str
    usability: str
    decode_tps: float | None
    ttft_ms: float | None
    p95_ms: float | None
    context: int | None
    reproductions: int
    result_id: str


def load_results(repo_root: Path) -> list[dict[str, Any]]:
    results = []
    for rel in RESULTS_DIRS:
        directory = repo_root / rel
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("schema_version") == "benchresult/v1":
                results.append(data)
    return results


def fold_matrix(results: list[dict[str, Any]]) -> list[MatrixRow]:
    def _rank(result: dict[str, Any]) -> tuple:
        return (
            1 if result.get("status") == "verified" else 0,
            len(result.get("reproductions") or []),
            result.get("measured_at") or "",
        )

    best: dict[tuple, dict[str, Any]] = {}
    for result in results:
        subject = result.get("subject") or {}
        key = (
            (result.get("pins") or {}).get("hwprofile_id") or "?",
            subject.get("modelprofile") or "?",
            subject.get("workload") or "?",
        )
        if key not in best or _rank(result) > _rank(best[key]):
            best[key] = result

    rows = []
    for (hwprofile, modelprofile, workload), result in sorted(best.items()):
        metrics = result.get("metrics") or {}
        rows.append(
            MatrixRow(
                hwprofile=hwprofile,
                modelprofile=modelprofile,
                workload=workload,
                status=result.get("status", "claimed"),
                usability=result.get("usability_suggested") or "unrated",
                decode_tps=metrics.get("decode_tps"),
                ttft_ms=metrics.get("ttft_ms"),
                p95_ms=(metrics.get("token_latency_ms") or {}).get("p95"),
                context=metrics.get("context_len_tokens"),
                reproductions=len(result.get("reproductions") or []),
                result_id=result.get("result_id", "?"),
            )
        )
    return rows


def render_markdown(rows: list[MatrixRow]) -> str:
    if not rows:
        return (
            "No benchmark results committed yet. Rows appear here as "
            "`benchresult/v1` files land in results/.\n"
        )
    def _fmt(value: Any) -> str:
        return "—" if value is None else str(value)

    lines = [
        "| Hardware | Model | Workload | Usability | Status | Decode tps | TTFT ms | p95 ms | Ctx | Repros |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.hwprofile} | {row.modelprofile} | {row.workload} "
            f"| {row.usability} | {row.status} | {_fmt(row.decode_tps)} "
            f"| {_fmt(row.ttft_ms)} | {_fmt(row.p95_ms)} | {_fmt(row.context)} "
            f"| {row.reproductions} |"
        )
    lines.append("")
    lines.append(
        "Usability labels are tool-suggested from documented thresholds; only "
        "`verified` rows (all pins + two reproductions) are release-grade."
    )
    return "\n".join(lines) + "\n"
