"""The fleet layer (RFC 0005): registry loading, fleet-wide planning, and
thin SSH wrappers around the existing CLI.

Remote execution is deliberately minimal: run the same `frontier` commands the
operator would run by hand in the remote checkout, and copy artifacts back for
review. No daemon, no state on the remote beyond the repo.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from frontier_bridge.planner.engine import PlanError, generate_plan

FLEET_DIRS = ("fleet/local", "fleet")
DEFAULT_WORKDIR = "~/Frontier-Bridge"
_VERDICT_ORDER = {"recommended": 0, "experimental": 1, "not_recommended": 2}


@dataclass
class Machine:
    name: str
    hwprofile: str | None
    ssh: str | None
    workdir: str
    roles: list[str]


@dataclass
class FleetPlanRow:
    machine: str
    hwprofile: str | None
    verdict: str
    quant: str | None
    reasons: list[str]


def load_fleet(repo_root: Path, fleet_file: Path | None = None) -> list[Machine]:
    """Load the fleet registry. Explicit file wins; else the first fleet/v1
    document found in fleet/local/ then fleet/."""
    candidates: list[Path] = []
    if fleet_file is not None:
        candidates = [fleet_file]
    else:
        for rel in FLEET_DIRS:
            directory = repo_root / rel
            if directory.is_dir():
                candidates.extend(sorted(directory.glob("*.yaml")))
    for path in candidates:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema_version") == "fleet/v1":
            return [
                Machine(
                    name=m["name"],
                    hwprofile=m.get("hwprofile"),
                    ssh=(m.get("reach") or {}).get("ssh"),
                    workdir=(m.get("reach") or {}).get("workdir") or DEFAULT_WORKDIR,
                    roles=m.get("roles") or [],
                )
                for m in data.get("machines", [])
            ]
    return []


def fleet_plan(
    repo_root: Path,
    machines: list[Machine],
    model_id: str,
    workload: str,
    context_budget: int,
    quant: str | None = None,
) -> list[FleetPlanRow]:
    """Run the existing planner against every registered machine's committed
    profile. Ranking is by verdict class only — no new scoring math."""
    rows = []
    for machine in machines:
        if not machine.hwprofile:
            rows.append(
                FleetPlanRow(machine.name, None, "unprofiled", None, ["no committed hwprofile"])
            )
            continue
        try:
            plan = generate_plan(
                repo_root=repo_root,
                model_id=model_id,
                hardware_id=machine.hwprofile,
                workload=workload,
                context_budget=context_budget,
                quant=quant,
            )
        except PlanError as exc:
            rows.append(FleetPlanRow(machine.name, machine.hwprofile, "error", None, [str(exc)]))
            continue
        chosen_quant = (plan.get("inputs") or {}).get("modelprofile", "").partition("/")[2] or None
        rows.append(
            FleetPlanRow(
                machine=machine.name,
                hwprofile=machine.hwprofile,
                verdict=plan.get("verdict", "?"),
                quant=chosen_quant,
                reasons=plan.get("reasons") or [],
            )
        )
    rows.sort(key=lambda r: _VERDICT_ORDER.get(r.verdict, 99))
    return rows


def _run(cmd: list[str], echo: Callable[[str], Any]) -> int:
    echo("$ " + " ".join(cmd))
    return subprocess.run(cmd).returncode


def remote_detect(
    machine: Machine,
    repo_root: Path,
    echo: Callable[[str], Any],
    skip_disk_bench: bool = False,
) -> Path | None:
    """Run `frontier detect` on the remote and pull the profile back into
    hardware_profiles/local/ for review before commit."""
    if not machine.ssh:
        echo(f"error: machine {machine.name!r} has no ssh reach")
        return None
    remote_out = f"{machine.workdir}/hardware_profiles/local/{machine.name}_detected.yaml"
    flags = " --skip-disk-bench" if skip_disk_bench else ""
    rc = _run(
        [
            "ssh", machine.ssh,
            f"cd {machine.workdir} && "
            f"(test -x .venv/bin/frontier && FRONTIER=.venv/bin/frontier || FRONTIER=frontier); "
            f"$FRONTIER detect -o {remote_out}{flags}",
        ],
        echo,
    )
    if rc != 0:
        echo(f"error: remote detect failed (exit {rc})")
        return None
    local_dir = repo_root / "hardware_profiles" / "local"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / f"{machine.name}_detected.yaml"
    rc = _run(["scp", f"{machine.ssh}:{remote_out}", str(local_path)], echo)
    if rc != 0:
        echo(f"error: could not copy the profile back (exit {rc})")
        return None
    return local_path


def remote_bench(
    machine: Machine,
    repo_root: Path,
    bench_args: list[str],
    echo: Callable[[str], Any],
) -> Path | None:
    """Run `frontier bench <args>` on the remote and pull results/local back."""
    if not machine.ssh:
        echo(f"error: machine {machine.name!r} has no ssh reach")
        return None
    joined = " ".join(bench_args)
    rc = _run(
        [
            "ssh", machine.ssh,
            f"cd {machine.workdir} && "
            f"(test -x .venv/bin/frontier && FRONTIER=.venv/bin/frontier || FRONTIER=frontier); "
            f"$FRONTIER bench {joined}",
        ],
        echo,
    )
    if rc != 0:
        echo(f"error: remote bench failed (exit {rc})")
        return None
    local_dir = repo_root / "results" / "local" / machine.name
    local_dir.mkdir(parents=True, exist_ok=True)
    rc = _run(
        ["scp", "-r", f"{machine.ssh}:{machine.workdir}/results/local/.", str(local_dir)],
        echo,
    )
    if rc != 0:
        echo(f"error: could not copy results back (exit {rc})")
        return None
    return local_dir
