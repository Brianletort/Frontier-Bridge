"""`frontier doctor`: is this machine ready?

A readiness diagnostic run before anything downloads. Every check reports
ok / warn / fail with the exact command that fixes it. Doctor never installs
anything itself — it tells you what is missing and how to get it, then (by
default) profiles the machine and matches it against committed runbooks so a
fresh seat ends its first minute knowing what it is and what it can run.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Below this much free space no catalog artifact fits (smallest is ~60 GB;
# the v0.1 targets start at 107 GB).
_FAIL_FREE_GB = 60.0
# Comfortable working space for a v0.1 target artifact plus room to work.
_WARN_FREE_GB = 150.0

OK = "ok"
WARN = "warn"
FAIL = "fail"


@dataclass
class CheckResult:
    name: str
    status: str  # ok | warn | fail
    detail: str
    fix: str | None = None


def _which_check(
    name: str,
    binary: str,
    which: Callable[[str], str | None],
    missing_status: str,
    detail_present: str,
    detail_missing: str,
    fix: str,
) -> CheckResult:
    path = which(binary)
    if path:
        return CheckResult(name, OK, detail_present.format(path=path))
    return CheckResult(name, missing_status, detail_missing, fix)


def run_checks(
    models_dest: Path,
    system: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    disk_free_gb: float | None = None,
) -> list[CheckResult]:
    """Run all environment checks. `system`, `which`, and `disk_free_gb` are
    injectable for tests; defaults inspect the real machine."""
    system = system or platform.system()
    results: list[CheckResult] = []

    version = sys.version_info
    results.append(
        CheckResult(
            "python",
            OK if version >= (3, 10) else FAIL,
            f"Python {version.major}.{version.minor}.{version.micro}",
            None if version >= (3, 10) else "install Python 3.10+ (pyenv or your distro packages)",
        )
    )

    results.append(
        _which_check(
            "git", "git", which, FAIL,
            "git at {path}",
            "git not found — cloning and DCO-signed contributions need it",
            "sudo apt install -y git" if system == "Linux" else "xcode-select --install",
        )
    )

    results.append(
        _which_check(
            "fio", "fio", which, WARN,
            "fio at {path} (detect records tool versions with measurements)",
            "fio not found — detect falls back to a Python read bench",
            "sudo apt install -y fio" if system == "Linux" else "brew install fio",
        )
    )

    if system == "Linux":
        results.append(
            _which_check(
                "nvidia-smi", "nvidia-smi", which, WARN,
                "nvidia-smi at {path}",
                "nvidia-smi not found — GPU nodes will be absent from the profile "
                "(fine on non-NVIDIA machines)",
                "install the NVIDIA driver for your distro, then re-run",
            )
        )
        results.append(
            _which_check(
                "boltctl", "boltctl", which, WARN,
                "boltctl at {path} (Thunderbolt device enumeration available)",
                "boltctl not found — only needed for eGPU / Thunderbolt topologies",
                "sudo apt install -y bolt",
            )
        )

    if disk_free_gb is None:
        probe = models_dest if models_dest.exists() else Path.cwd()
        disk_free_gb = shutil.disk_usage(probe).free / 1e9
    if disk_free_gb < _FAIL_FREE_GB:
        status, fix = FAIL, "free up disk space or pass a different --dest volume"
    elif disk_free_gb < _WARN_FREE_GB:
        status, fix = WARN, (
            "the v0.1 target artifacts are 107-467 GB; plan a destination with room "
            "(frontier setup free-space-checks before downloading)"
        )
    else:
        status, fix = OK, None
    results.append(
        CheckResult(
            "disk-space",
            status,
            f"{disk_free_gb:.0f} GB free at {models_dest if models_dest.exists() else Path.cwd()}",
            fix,
        )
    )

    llama_fix = (
        "scripts/build_llama_cpp_cuda.sh (CUDA build, pinned commit)"
        if system == "Linux"
        else "brew install llama.cpp"
    )
    results.append(
        _which_check(
            "llama-server", "llama-server", which, WARN,
            "llama-server at {path}",
            "llama-server not found — needed before frontier run/bench with the llama_cpp engine",
            llama_fix,
        )
    )
    results.append(
        _which_check(
            "ssh", "ssh", which, WARN,
            "ssh at {path} (fleet detect/bench can drive remote machines)",
            "ssh not found — frontier fleet remote verbs unavailable",
            "sudo apt install -y openssh-client" if system == "Linux" else "part of macOS",
        )
    )
    return results


def nvidia_summary() -> str | None:
    """One-line GPU summary when nvidia-smi is present; None otherwise."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return "; ".join(lines) if lines else None


def worst_status(results: list[CheckResult]) -> str:
    statuses = {r.status for r in results}
    if FAIL in statuses:
        return FAIL
    if WARN in statuses:
        return WARN
    return OK
