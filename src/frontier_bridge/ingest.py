"""Ingest Hugging Face GGUF repos into modelprofile/v1 documents.

The tool fetches the repo file tree from the HF API, groups GGUF shards into
quant artifacts, pins sha256 from the LFS metadata (the API's `lfs.oid` is the
file's sha256), and optionally range-inspects the GGUF headers to measure the
memory model (dense vs routed-expert bytes) and architecture facts (param
count, expert counts, context length).

Everything in the emitted profile is either measured (headers, LFS pins, file
sizes) or copied from upstream metadata and marked as claimed. Nothing is
guessed; unknown values stay null. Generated profiles are drafts for human
review before commit.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from frontier_bridge.gguf import inspect_artifact

HF_BASE = "https://huggingface.co"

# Quant token inside a GGUF filename, e.g. Q4_K_M, UD-Q2_K_XL, IQ4_XS, BF16.
_QUANT_TOKEN = re.compile(
    r"(?i)\b((?:UD-)?(?:I?Q\d(?:[_-][A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:[_-][A-Z0-9]+)*))\b"
)
_SHARD_SUFFIX = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


class IngestError(Exception):
    """Repo unreachable, no GGUF files, or ambiguous quant selection."""


@dataclass
class QuantGroup:
    """One quant variant of a repo: its shards in order."""

    upstream_name: str
    files: list[dict[str, Any]] = field(default_factory=list)

    @property
    def quant_id(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.upstream_name.lower()).strip("_")

    @property
    def size_gb(self) -> float:
        return round(sum(f.get("size") or 0 for f in self.files) / 1e9, 1)


def _http_get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "frontier-bridge"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_repo(repo: str) -> tuple[str, str]:
    """Accept `org/name`, a full HF URL, or a /tree/<ref> URL. Returns (repo_id, ref)."""
    ref = "main"
    cleaned = repo.strip().rstrip("/")
    if cleaned.startswith(("http://", "https://")):
        cleaned = cleaned.split("://", 1)[1]
        cleaned = cleaned.split("/", 1)[1] if "/" in cleaned else ""
    if "/tree/" in cleaned:
        cleaned, _, rest = cleaned.partition("/tree/")
        ref = rest.split("/", 1)[0] or "main"
    parts = [p for p in cleaned.split("/") if p]
    if len(parts) < 2:
        raise IngestError(f"cannot parse HF repo from {repo!r} (expected org/name)")
    return "/".join(parts[:2]), ref


def fetch_tree(
    repo_id: str, ref: str = "main", fetch: Callable[[str], Any] = _http_get_json
) -> list[dict[str, Any]]:
    """All files in the repo (recursive listing from the HF API)."""
    url = f"{HF_BASE}/api/models/{repo_id}/tree/{ref}?recursive=true"
    try:
        entries = fetch(url)
    except OSError as exc:
        raise IngestError(f"cannot list {repo_id}@{ref}: {exc}") from exc
    if not isinstance(entries, list):
        raise IngestError(f"unexpected HF API response for {repo_id}")
    return [e for e in entries if e.get("type") == "file"]


def fetch_repo_meta(
    repo_id: str, fetch: Callable[[str], Any] = _http_get_json
) -> dict[str, Any]:
    """Repo-level metadata (license lives in cardData). Missing data is not fatal."""
    try:
        meta = fetch(f"{HF_BASE}/api/models/{repo_id}")
    except OSError:
        return {}
    return meta if isinstance(meta, dict) else {}


def group_quants(files: list[dict[str, Any]]) -> list[QuantGroup]:
    """Group GGUF files into quant variants.

    Files inside a subdirectory take the top-level directory name as the quant;
    flat files are grouped by the quant token in the filename.
    """
    groups: dict[str, QuantGroup] = {}
    for entry in files:
        path = entry.get("path") or ""
        if not path.lower().endswith(".gguf"):
            continue
        if "/" in path:
            name = path.split("/", 1)[0]
        else:
            match = _QUANT_TOKEN.search(_SHARD_SUFFIX.sub(".gguf", path))
            name = match.group(1) if match else "unknown"
        groups.setdefault(name, QuantGroup(upstream_name=name)).files.append(entry)
    for group in groups.values():
        group.files.sort(key=lambda f: f.get("path") or "")
    return sorted(groups.values(), key=lambda g: g.upstream_name)


def select_quant(groups: list[QuantGroup], wanted: str | None) -> QuantGroup:
    if not groups:
        raise IngestError("no GGUF files found in this repo")
    if wanted is None:
        if len(groups) == 1:
            return groups[0]
        names = ", ".join(g.upstream_name for g in groups)
        raise IngestError(f"repo has multiple quants ({names}); pass --quant")
    for group in groups:
        if wanted.lower() in (group.upstream_name.lower(), group.quant_id):
            return group
    names = ", ".join(g.upstream_name for g in groups)
    raise IngestError(f"quant {wanted!r} not found; available: {names}")


def shard_urls(repo_id: str, ref: str, group: QuantGroup) -> list[str]:
    return [f"{HF_BASE}/{repo_id}/resolve/{ref}/{f['path']}" for f in group.files]


def _sha256_of(entry: dict[str, Any]) -> str | None:
    lfs = entry.get("lfs") or {}
    oid = lfs.get("oid")
    # HF LFS oids are the file's sha256 (64 hex chars).
    if isinstance(oid, str) and len(oid) == 64:
        return oid
    return None


def build_profile(
    repo_id: str,
    ref: str,
    group: QuantGroup,
    model_id: str,
    family: str | None = None,
    license_name: str | None = None,
    inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a modelprofile/v1 draft from API metadata and header inspection."""
    quant = group.quant_id
    shards = [
        {"path": f["path"], "sha256": _sha256_of(f)}
        for f in group.files
    ]
    source_dir = group.files[0]["path"].rsplit("/", 1)[0] if "/" in group.files[0]["path"] else ""
    source = f"{HF_BASE}/{repo_id}/tree/{ref}" + (f"/{source_dir}" if source_dir else "")

    architecture: dict[str, Any] = {
        "type": "dense",
        "params_total_b": None,
        "params_active_b": None,
        "experts": {"routed_total": None, "active_per_token": None, "shared": None},
        "attention": {"type": None, "kv_scaling": None},
        "context_max": None,
        "spec_decode": {"mtp": "unknown"},
    }
    memory_model: dict[str, Any] = {
        "dense_resident_gb": {quant: None},
        "per_expert_gb": {quant: None},
        "kv_per_1k_tokens_mb": None,
    }
    if inspection:
        if inspection.get("expert_count"):
            architecture["type"] = "moe"
            architecture["experts"] = {
                "routed_total": inspection.get("expert_count"),
                "active_per_token": inspection.get("expert_used_count"),
                "shared": inspection.get("expert_shared_count"),
            }
        architecture["params_total_b"] = inspection.get("params_total_b")
        architecture["context_max"] = inspection.get("context_length")
        memory_model["dense_resident_gb"][quant] = inspection.get("dense_resident_gb")
        memory_model["per_expert_gb"][quant] = inspection.get("per_expert_layer_gb")
        memory_model["measurement"] = {
            "method": inspection.get("method"),
            "routed_experts_gb": {quant: inspection.get("routed_experts_gb")},
        }

    artifact: dict[str, Any] = {
        "format": "gguf",
        "quant": quant,
        "upstream_name": group.upstream_name,
        "size_gb": group.size_gb,
        "source": source,
        "sha256": shards[0]["sha256"] if shards else None,
    }
    if len(shards) > 1 or (shards and "/" in shards[0]["path"]):
        artifact["shards"] = shards

    return {
        "schema_version": "modelprofile/v1",
        "model_id": model_id,
        "family": family or model_id.split("-")[0],
        "license": {"name": license_name, "commercial_ok": "unknown"},
        "architecture": architecture,
        "artifacts": [artifact],
        "runtime_support": [
            {
                "runtime": "llama_cpp",
                "status": "claimed",
                "source": f"{HF_BASE}/{repo_id}",
            }
        ],
        "memory_model": memory_model,
        "known_failure_modes": [],
    }


def ingest_repo(
    repo: str,
    model_id: str | None = None,
    quant: str | None = None,
    family: str | None = None,
    inspect_headers: bool = True,
    fetch: Callable[[str], Any] = _http_get_json,
) -> dict[str, Any]:
    """End-to-end ingest: tree -> quant group -> (optional) header inspection -> profile."""
    repo_id, ref = parse_repo(repo)
    files = fetch_tree(repo_id, ref, fetch=fetch)
    group = select_quant(group_quants(files), quant)

    if model_id is None:
        # Derive from the repo name: strip a trailing -GGUF marker, lowercase.
        raw = repo_id.split("/", 1)[1]
        raw = re.sub(r"(?i)[-_.]gguf$", "", raw)
        model_id = re.sub(r"[^a-z0-9._-]+", "-", raw.lower()).strip("-.")

    meta = fetch_repo_meta(repo_id, fetch=fetch)
    license_name = (meta.get("cardData") or {}).get("license")
    if isinstance(license_name, list):
        license_name = license_name[0] if license_name else None

    inspection = None
    if inspect_headers:
        inspection = inspect_artifact(shard_urls(repo_id, ref, group))

    return build_profile(
        repo_id,
        ref,
        group,
        model_id=model_id,
        family=family,
        license_name=license_name,
        inspection=inspection,
    )
