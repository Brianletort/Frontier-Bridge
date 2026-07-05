"""`frontier run`: execute a plan's launch command and health-check the endpoint.

Thin wrapper by design — the runtime does the work. This module:
1. substitutes the artifact path into the plan's recorded launch command,
2. optionally verifies the artifact sha256 against the model profile pins,
3. launches the runtime as a subprocess, streaming its output,
4. polls the OpenAI-compatible endpoint until it responds or times out.

Commands are printed before execution, never run silently.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_PATH_PLACEHOLDERS = ("<GGUF_PATH>", "<MODEL_PATH>")
_HASH_CHUNK = 8 * 1024 * 1024


class RunError(Exception):
    """Unrecoverable run setup error (bad plan, missing artifact, hash mismatch)."""


@dataclass
class LaunchSpec:
    command: str
    args: list[str]
    port: int
    engine: str


def sha256_file(path: Path, progress: Callable[[int], None] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            if progress:
                progress(len(chunk))
    return digest.hexdigest()


def verify_artifact(model_path: Path, model_profile: dict[str, Any], quant: str) -> str:
    """Verify the artifact file (or its first shard) against the profile pins.

    Returns the matched pin description. Raises RunError on mismatch or when
    the profile has no pin for this file.
    """
    artifact = next(
        (a for a in model_profile.get("artifacts", []) if a.get("quant") == quant),
        None,
    )
    if artifact is None:
        raise RunError(f"model profile has no artifact for quant {quant!r}")

    pinned: dict[str, str] = {}
    if artifact.get("sha256"):
        pinned[Path(artifact.get("source") or "artifact").name] = artifact["sha256"]
    for shard in artifact.get("shards") or []:
        if shard.get("sha256") and shard.get("path"):
            pinned[Path(shard["path"]).name] = shard["sha256"]
    if not pinned:
        raise RunError(
            f"no sha256 pins recorded for {quant!r} — pin hashes before verifying"
        )

    expected = pinned.get(model_path.name)
    if expected is None:
        raise RunError(
            f"{model_path.name} does not match any pinned shard name; "
            f"pinned: {sorted(pinned)}"
        )
    actual = sha256_file(model_path)
    if actual != expected:
        raise RunError(
            f"sha256 mismatch for {model_path.name}:\n"
            f"  expected {expected}\n  actual   {actual}\n"
            "Do not run unverified artifacts."
        )
    return f"{model_path.name} sha256 verified"


def build_launch(plan: dict[str, Any], model_path: str) -> LaunchSpec:
    """Substitute the artifact path into the plan's recorded launch command."""
    runtime = plan.get("runtime") or {}
    command: str = runtime.get("launch") or ""
    if not command:
        raise RunError("plan has no runtime.launch command (refusal plan?)")
    command = command.split("#", 1)[0].strip()  # drop the advisory comment
    for placeholder in _PATH_PLACEHOLDERS:
        command = command.replace(placeholder, model_path)
    args = shlex.split(command)
    port = 8080
    for i, arg in enumerate(args):
        if arg in ("--port", "-p") and i + 1 < len(args) and args[i + 1].isdigit():
            port = int(args[i + 1])
    return LaunchSpec(command=command, args=args, port=port, engine=runtime.get("engine", "?"))


def wait_for_endpoint(port: int, timeout_s: float = 600.0, interval_s: float = 2.0) -> bool:
    """Poll the OpenAI-compatible /v1/models endpoint until it answers."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(interval_s)
    return False


def probe_endpoint(port: int) -> dict[str, Any] | None:
    """One-shot health probe; returns the parsed /v1/models payload or None."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/models", timeout=5
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
        return None


def launch_and_wait(
    spec: LaunchSpec,
    echo: Callable[[str], None],
    ready_timeout_s: float = 600.0,
) -> subprocess.Popen:
    """Start the runtime and block until the endpoint is up (or raise)."""
    echo(f"Launching: {spec.command}")
    process = subprocess.Popen(spec.args)
    echo(f"Runtime pid {process.pid}; waiting for http://127.0.0.1:{spec.port}/v1/models ...")
    if not wait_for_endpoint(spec.port, timeout_s=ready_timeout_s):
        process.terminate()
        raise RunError(
            f"endpoint did not become ready within {int(ready_timeout_s)}s; "
            "runtime terminated"
        )
    echo("Endpoint ready.")
    return process