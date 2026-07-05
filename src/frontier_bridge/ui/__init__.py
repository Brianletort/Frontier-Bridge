"""`frontier ui`: a local web app over the same library the CLI uses.

Stdlib-only by design (ThreadingHTTPServer + a static single-page frontend) so
the UI adds zero dependencies. Endpoints are thin wrappers over catalog,
planner, provisioning, and results code — the CLI remains the source of truth.

JSON API:
    GET  /api/state            hardware + models + matrix in one call
    POST /api/detect           run hardware detection on this machine
    GET  /api/plan             query params: model, hardware, workload, ctx, quant
    POST /api/setup            start a background setup job (download + launch)
    GET  /api/job              current job status + log tail
    POST /api/job/stop         stop the running runtime
    GET  /api/endpoint?port=   probe a running OpenAI-compatible endpoint
"""

from __future__ import annotations

import json
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any

from frontier_bridge.catalog import (
    find_repo_root,
    get_model_profiles,
    list_hardware_profiles,
    list_model_profiles,
)
from frontier_bridge.detect import detect_hardware
from frontier_bridge.planner.engine import PlanError, generate_plan
from frontier_bridge.provision import SetupError, download_artifact, ensure_runtime
from frontier_bridge.results import fold_matrix, load_results
from frontier_bridge.runner import RunError, build_launch, launch_and_wait, probe_endpoint


class SetupJob:
    """One provisioning job at a time: download -> launch, with a live log."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = "idle"  # idle | running | ready | failed | stopped
        self.log: list[str] = []
        self.process = None
        self.port: int | None = None
        self._thread: threading.Thread | None = None

    def echo(self, line: str) -> None:
        with self._lock:
            self.log.append(line)
            if len(self.log) > 500:
                del self.log[: len(self.log) - 500]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"status": self.status, "log": list(self.log[-100:]), "port": self.port}

    def start(self, repo_root: Path, plan: dict[str, Any], dest: Path, launch: bool) -> bool:
        with self._lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.log = []
        self._thread = threading.Thread(
            target=self._run, args=(repo_root, plan, dest, launch), daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> bool:
        process = self.process
        if process is None or process.poll() is not None:
            return False
        self.echo("Stopping runtime...")
        process.terminate()
        try:
            process.wait(timeout=30)
        except Exception:
            process.kill()
        self.status = "stopped"
        self.process = None
        return True

    def _run(self, repo_root: Path, plan: dict[str, Any], dest: Path, launch: bool) -> None:
        try:
            engine = (plan.get("runtime") or {}).get("engine") or "?"
            model_id, _, quant = (plan.get("inputs", {}).get("modelprofile", "")).partition("/")
            self.echo(f"[1/3] Runtime: {engine}")
            if not ensure_runtime(engine, echo=self.echo):
                self.status = "failed"
                return
            self.echo(f"[2/3] Artifact: {model_id}/{quant} -> {dest}")
            profile = next(
                (
                    e.data
                    for e in get_model_profiles(repo_root, model_id)
                    if any(a.get("quant") == quant for a in e.data.get("artifacts", []))
                ),
                None,
            )
            if profile is None:
                self.echo(f"error: no model profile for {model_id}/{quant}")
                self.status = "failed"
                return
            model_path = download_artifact(profile, quant, dest, echo=self.echo)
            self.echo(f"Artifact ready: {model_path}")
            if not launch:
                self.echo("[3/3] Launch skipped.")
                self.status = "ready"
                return
            self.echo("[3/3] Launching...")
            spec = build_launch(plan, str(model_path))
            self.process = launch_and_wait(spec, echo=self.echo, ready_timeout_s=600)
            self.port = spec.port
            self.echo(f"Endpoint live at http://127.0.0.1:{spec.port}/v1")
            self.status = "ready"
        except (SetupError, RunError, OSError) as exc:
            self.echo(f"error: {exc}")
            self.status = "failed"


def _state(repo_root: Path) -> dict[str, Any]:
    hardware = [
        {
            "profile_id": h.profile_id,
            "method": h.method,
            "summary": h.summary,
        }
        for h in list_hardware_profiles(repo_root)
    ]
    models: dict[str, dict[str, Any]] = {}
    for m in list_model_profiles(repo_root):
        entry = models.setdefault(
            m.model_id,
            {"model_id": m.model_id, "summary": m.summary, "quants": []},
        )
        for artifact in m.data.get("artifacts", []):
            entry["quants"].append(
                {
                    "quant": artifact.get("quant"),
                    "size_gb": artifact.get("size_gb"),
                    "pinned": bool(artifact.get("sha256")),
                }
            )
    matrix = [
        {
            "hwprofile": r.hwprofile,
            "modelprofile": r.modelprofile,
            "workload": r.workload,
            "status": r.status,
            "usability": r.usability,
            "decode_tps": r.decode_tps,
            "ttft_ms": r.ttft_ms,
            "context": r.context,
            "reproductions": r.reproductions,
        }
        for r in fold_matrix(load_results(repo_root))
    ]
    return {
        "hardware": hardware,
        "models": sorted(models.values(), key=lambda m: m["model_id"]),
        "matrix": matrix,
    }


def make_handler(repo_root: Path, job: SetupJob):
    """Build the request-handler class bound to this repo root and job."""

    index_html = (resources.files("frontier_bridge") / "ui" / "static" / "index.html").read_text(
        encoding="utf-8"
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet server
            pass

        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
            if parsed.path == "/":
                self._send_html(index_html)
            elif parsed.path == "/api/state":
                self._send_json(_state(repo_root))
            elif parsed.path == "/api/plan":
                try:
                    plan = generate_plan(
                        repo_root=repo_root,
                        model_id=query.get("model", ""),
                        hardware_id=query.get("hardware", ""),
                        workload=query.get("workload", "chat"),
                        context_budget=int(query.get("ctx", 32768)),
                        quant=query.get("quant") or None,
                        engine_override=query.get("engine") or None,
                    )
                    self._send_json(plan)
                except (PlanError, ValueError) as exc:
                    self._send_json({"error": str(exc)}, status=400)
            elif parsed.path == "/api/job":
                self._send_json(job.snapshot())
            elif parsed.path == "/api/endpoint":
                try:
                    port = int(query.get("port", 8080))
                except ValueError:
                    self._send_json({"error": "bad port"}, status=400)
                    return
                info = probe_endpoint(port)
                self._send_json({"up": info is not None, "models": info})
            else:
                self._send_json({"error": "not found"}, status=404)

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/detect":
                body = self._read_body()
                try:
                    profile = detect_hardware(
                        run_disk_bench=bool(body.get("disk_bench", False))
                    )
                    self._send_json(profile)
                except NotImplementedError as exc:
                    self._send_json({"error": str(exc)}, status=400)
            elif parsed.path == "/api/setup":
                body = self._read_body()
                plan = body.get("plan")
                if not isinstance(plan, dict) or plan.get("schema_version") != "plan/v1":
                    self._send_json({"error": "body.plan must be a plan/v1 object"}, status=400)
                    return
                if plan.get("verdict") == "not_recommended":
                    self._send_json({"error": "refusal plans cannot be set up"}, status=400)
                    return
                dest = Path(body.get("dest") or (repo_root / "models"))
                started = job.start(
                    repo_root, plan, dest, launch=bool(body.get("launch", True))
                )
                if not started:
                    self._send_json({"error": "a setup job is already running"}, status=409)
                    return
                self._send_json({"ok": True})
            elif parsed.path == "/api/job/stop":
                self._send_json({"stopped": job.stop()})
            else:
                self._send_json({"error": "not found"}, status=404)

    return Handler


def make_server(repo_root: Path | None = None, port: int = 7861) -> ThreadingHTTPServer:
    root = repo_root or find_repo_root()
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(root, SetupJob()))


def serve(port: int = 7861, open_browser: bool = True) -> None:
    server = make_server(port=port)
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}"
    print(f"Frontier Bridge UI: {url}  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
