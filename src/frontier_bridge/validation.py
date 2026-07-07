"""Schema loading and instance validation for the v1 schemas.

Files are matched to schemas by their ``schema_version`` field. Files without
one are skipped (they are not Frontier Bridge instances). Unknown schema
versions are an error: validators reject rather than guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

import jsonschema
import yaml

SCHEMA_FILES = {
    "hwprofile/v1": "hwprofile.v1.json",
    "modelprofile/v1": "modelprofile.v1.json",
    "plan/v1": "plan.v1.json",
    "benchresult/v1": "benchresult.v1.json",
    "runbook/v1": "runbook.v1.json",
    "fleet/v1": "fleet.v1.json",
}

_INSTANCE_SUFFIXES = {".yaml", ".yml", ".json"}
_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "schemas"}


@dataclass
class ValidationIssue:
    path: Path
    message: str


@dataclass
class ValidationReport:
    checked: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def load_schema(schema_version: str) -> dict[str, Any]:
    """Load a bundled JSON Schema by its schema_version string."""
    try:
        filename = SCHEMA_FILES[schema_version]
    except KeyError:
        raise KeyError(
            f"Unknown schema_version {schema_version!r}. "
            f"Known: {sorted(SCHEMA_FILES)}"
        ) from None
    ref = resources.files("frontier_bridge") / "schemas" / filename
    return json.loads(ref.read_text(encoding="utf-8"))


def load_instance(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def validate_instance(instance: dict[str, Any]) -> list[str]:
    """Validate one instance dict against its declared schema. Returns error messages."""
    schema_version = instance.get("schema_version")
    if not isinstance(schema_version, str):
        return ["missing or non-string 'schema_version' field"]
    try:
        schema = load_schema(schema_version)
    except KeyError as exc:
        return [str(exc)]
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    ]


def _iter_candidate_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix not in _INSTANCE_SUFFIXES:
            continue
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def validate_path(root: Path) -> ValidationReport:
    """Validate every YAML/JSON instance under root that declares a schema_version."""
    report = ValidationReport()
    for path in _iter_candidate_files(root):
        try:
            instance = load_instance(path)
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            report.checked.append(path)
            report.issues.append(ValidationIssue(path, f"parse error: {exc}"))
            continue
        if not isinstance(instance, dict) or "schema_version" not in instance:
            report.skipped.append(path)
            continue
        report.checked.append(path)
        for message in validate_instance(instance):
            report.issues.append(ValidationIssue(path, message))
    return report
