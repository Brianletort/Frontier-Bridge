"""Discovery of committed hardware and model profiles in the repository."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

HARDWARE_DIR = "hardware_profiles"
MODEL_DIR = "model_profiles"


@dataclass
class HardwareEntry:
    profile_id: str
    method: str
    summary: str
    path: Path
    data: dict


@dataclass
class ModelEntry:
    model_id: str
    quant: str
    summary: str
    path: Path
    data: dict


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from start (default: cwd) to the first directory containing profile dirs or .git."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / HARDWARE_DIR).is_dir() or (candidate / ".git").is_dir():
            return candidate
    return current


def _load_yaml_files(directory: Path) -> list[tuple[Path, dict]]:
    if not directory.is_dir():
        return []
    out: list[tuple[Path, dict]] = []
    for path in sorted(directory.rglob("*.yaml")):
        if "templates" in path.relative_to(directory).parts:
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            out.append((path, data))
    return out


def _memory_summary(data: dict) -> str:
    parts = []
    for node in data.get("nodes", []):
        if node.get("kind") == "memory":
            cap = node.get("capacity_gb")
            parts.append(f"{node.get('class')}={cap if cap is not None else '?'}GB")
        elif node.get("kind") == "storage":
            measured = (node.get("measured") or {}).get("seq_read_gbps")
            parts.append(f"{node.get('class')}@{measured if measured is not None else '?'}GB/s")
    return " ".join(parts) or "(no memory nodes)"


def list_hardware_profiles(repo_root: Path) -> list[HardwareEntry]:
    entries = []
    for path, data in _load_yaml_files(repo_root / HARDWARE_DIR):
        if data.get("schema_version") != "hwprofile/v1":
            continue
        entries.append(
            HardwareEntry(
                profile_id=data.get("profile_id", path.stem),
                method=(data.get("provenance") or {}).get("method", "unknown"),
                summary=_memory_summary(data),
                path=path,
                data=data,
            )
        )
    return entries


def list_model_profiles(repo_root: Path) -> list[ModelEntry]:
    entries = []
    for path, data in _load_yaml_files(repo_root / MODEL_DIR):
        if data.get("schema_version") != "modelprofile/v1":
            continue
        arch = data.get("architecture") or {}
        quants = ", ".join(a.get("quant", "?") for a in data.get("artifacts", []))
        total = arch.get("params_total_b")
        active = arch.get("params_active_b")
        summary = (
            f"{arch.get('type', '?')} "
            f"{total if total is not None else '?'}B total"
            f" / {active if active is not None else '?'}B active"
        )
        entries.append(
            ModelEntry(
                model_id=data.get("model_id", path.stem),
                quant=quants or "?",
                summary=summary,
                path=path,
                data=data,
            )
        )
    return entries


def get_hardware_profile(repo_root: Path, profile_id: str) -> HardwareEntry | None:
    for entry in list_hardware_profiles(repo_root):
        if entry.profile_id == profile_id:
            return entry
    return None


def get_model_profiles(repo_root: Path, model_id: str) -> list[ModelEntry]:
    return [e for e in list_model_profiles(repo_root) if e.model_id == model_id]
