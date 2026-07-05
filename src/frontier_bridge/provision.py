"""`frontier setup`: from a plan to a running endpoint on this machine.

Three responsibilities, all orchestration — no new runtime logic:

1. Runtime availability: check the plan's engine binary is on PATH; install
   llama.cpp via Homebrew when possible, otherwise print the documented
   install route. Nothing is installed silently.
2. Artifact download: resolve the pinned shards from the model profile,
   free-space check against the destination volume, download with resume
   (HTTP Range into a .part file), verify sha256 against the profile pin
   before the shard gets its final name. Unpinned shards are refused unless
   explicitly allowed.
3. Launch: hand the verified artifact to the existing runner path.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from frontier_bridge.runner import sha256_file

_DOWNLOAD_CHUNK = 8 * 1024 * 1024

# Engine -> (binary to look for, install command by platform, docs).
_RUNTIME_BINARIES: dict[str, str] = {
    "llama_cpp": "llama-server",
    "ds4": "ds4",
    "ds4_zgx": "ds4",
    "mlx": "mlx_lm.server",
    "vllm": "vllm",
    "sglang": "python",  # module launch; presence check is best-effort
}

_INSTALL_HINTS: dict[str, str] = {
    "llama_cpp": (
        "brew install llama.cpp  (macOS/Linuxbrew)\n"
        "  or download a release: https://github.com/ggml-org/llama.cpp/releases"
    ),
    "ds4": "see https://github.com/mitkox/ds4-zgx-gb10",
    "ds4_zgx": "see https://github.com/mitkox/ds4-zgx-gb10",
    "mlx": "pip install mlx-lm",
    "vllm": "pip install vllm",
    "sglang": "pip install 'sglang[all]'",
}


class SetupError(Exception):
    """Unrecoverable setup failure (no space, bad pin, runtime missing)."""


@dataclass
class ShardSpec:
    url: str
    filename: str
    sha256: str | None
    size_bytes: int | None = None


def runtime_binary(engine: str) -> str | None:
    return _RUNTIME_BINARIES.get(engine)


def runtime_available(engine: str) -> bool:
    binary = runtime_binary(engine)
    return bool(binary and shutil.which(binary))


def install_hint(engine: str) -> str:
    return _INSTALL_HINTS.get(engine, f"no documented install route for {engine!r}")


def ensure_runtime(
    engine: str,
    echo: Callable[[str], None],
    auto_install: bool = True,
) -> bool:
    """Make the engine's binary available. Returns True when ready.

    Only llama.cpp has a safe unattended install (Homebrew). Everything else
    gets its documented route printed and returns False.
    """
    if runtime_available(engine):
        return True
    if engine == "llama_cpp" and auto_install and shutil.which("brew"):
        echo("llama-server not found — installing llama.cpp via Homebrew...")
        result = subprocess.run(
            ["brew", "install", "llama.cpp"], capture_output=True, text=True
        )
        if result.returncode != 0:
            echo(f"brew install failed:\n{result.stderr.strip()}")
            return False
        return runtime_available(engine)
    echo(f"Runtime {engine!r} is not installed. Install it with:\n  {install_hint(engine)}")
    return False


def resolve_shards(model_profile: dict[str, Any], quant: str) -> list[ShardSpec]:
    """Pinned shard download specs for the artifact backing this quant."""
    artifact = next(
        (a for a in model_profile.get("artifacts", []) if a.get("quant") == quant),
        None,
    )
    if artifact is None:
        raise SetupError(f"model profile has no artifact for quant {quant!r}")
    source = artifact.get("source") or ""
    shards = artifact.get("shards") or []
    specs: list[ShardSpec] = []
    if shards and "/tree/" in source:
        base, _, ref_and_dir = source.partition("/tree/")
        ref = ref_and_dir.split("/", 1)[0]
        for shard in shards:
            path = shard.get("path")
            if not path:
                continue
            specs.append(
                ShardSpec(
                    url=f"{base}/resolve/{ref}/{path}",
                    filename=Path(path).name,
                    sha256=shard.get("sha256"),
                )
            )
    elif source:
        specs.append(
            ShardSpec(
                url=source,
                filename=Path(source).name or "model.gguf",
                sha256=artifact.get("sha256"),
            )
        )
    if not specs:
        raise SetupError(
            f"artifact for {quant!r} has no downloadable source; "
            "add shard paths or a direct source URL to the model profile"
        )
    return specs


def check_free_space(dest_dir: Path, required_gb: float | None) -> None:
    """Refuse before downloading into a full disk. Needs artifact size + 5% slack."""
    if required_gb is None:
        return
    free_gb = shutil.disk_usage(dest_dir).free / 1e9
    needed = required_gb * 1.05
    if free_gb < needed:
        raise SetupError(
            f"not enough free space in {dest_dir}: {free_gb:.1f}GB free, "
            f"~{needed:.1f}GB needed (artifact {required_gb}GB + 5% slack)"
        )


def download_shard(
    spec: ShardSpec,
    dest_dir: Path,
    echo: Callable[[str], None],
    verify: bool = True,
    allow_unpinned: bool = False,
) -> Path:
    """Download one shard with resume, then verify its sha256 pin.

    The file lands as `<name>.part` and is renamed only after the hash check
    passes — a completed filename always means a verified artifact.
    """
    if spec.sha256 is None and verify and not allow_unpinned:
        raise SetupError(
            f"{spec.filename} has no sha256 pin in the model profile; "
            "refusing to download unverifiable weights (--allow-unpinned to override)"
        )
    final = dest_dir / spec.filename
    if final.exists():
        if not verify or spec.sha256 is None:
            echo(f"  {spec.filename}: already present (hash not checked)")
            return final
        echo(f"  {spec.filename}: already present, verifying...")
        if sha256_file(final) == spec.sha256:
            return final
        raise SetupError(
            f"{final} exists but does not match its pin — move it away and retry"
        )

    part = dest_dir / (spec.filename + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "frontier-bridge"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        echo(f"  {spec.filename}: resuming at {offset / 1e9:.2f}GB")
    else:
        echo(f"  {spec.filename}: downloading...")

    request = urllib.request.Request(spec.url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if offset and response.status != 206:
                # Server ignored the Range header: start over.
                offset = 0
            mode = "ab" if offset else "wb"
            done = offset
            last_report = time.monotonic()
            with part.open(mode) as f:
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if now - last_report > 10:
                        echo(f"    {spec.filename}: {done / 1e9:.2f}GB")
                        last_report = now
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise SetupError(
            f"download of {spec.filename} failed at {part.stat().st_size if part.exists() else 0} "
            f"bytes: {exc} — rerun to resume"
        ) from exc

    if verify and spec.sha256 is not None:
        echo(f"  {spec.filename}: verifying sha256...")
        actual = sha256_file(part)
        if actual != spec.sha256:
            raise SetupError(
                f"sha256 mismatch for {spec.filename}:\n"
                f"  expected {spec.sha256}\n  actual   {actual}\n"
                "The .part file was kept for inspection; do not run it."
            )
    part.rename(final)
    return final


def download_artifact(
    model_profile: dict[str, Any],
    quant: str,
    dest_dir: Path,
    echo: Callable[[str], None],
    verify: bool = True,
    allow_unpinned: bool = False,
) -> Path:
    """Download all shards of an artifact. Returns the primary (first) shard path."""
    specs = resolve_shards(model_profile, quant)
    dest_dir.mkdir(parents=True, exist_ok=True)
    artifact = next(
        a for a in model_profile.get("artifacts", []) if a.get("quant") == quant
    )
    already = sum(
        (dest_dir / s.filename).stat().st_size
        for s in specs
        if (dest_dir / s.filename).exists()
    )
    required_gb = artifact.get("size_gb")
    if required_gb is not None:
        required_gb = max(required_gb - already / 1e9, 0)
    check_free_space(dest_dir, required_gb)

    paths = [
        download_shard(
            spec, dest_dir, echo=echo, verify=verify, allow_unpinned=allow_unpinned
        )
        for spec in specs
    ]
    return paths[0]


__all__ = [
    "SetupError",
    "ShardSpec",
    "check_free_space",
    "download_artifact",
    "download_shard",
    "ensure_runtime",
    "install_hint",
    "resolve_shards",
    "runtime_available",
    "runtime_binary",
]
