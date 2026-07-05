"""The `frontier` command-line interface."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import yaml

from frontier_bridge import __version__
from frontier_bridge.catalog import (
    find_repo_root,
    get_model_profiles,
    list_hardware_profiles,
    list_model_profiles,
)
from frontier_bridge.bench.collectors import start_collectors, stop_and_report
from frontier_bridge.bench.engine import SUITES, build_result, load_suite, plan_hash, run_suite
from frontier_bridge.detect import detect_hardware
from frontier_bridge.gguf import GGUFError, inspect_artifact
from frontier_bridge.planner.engine import PlanError, generate_plan
from frontier_bridge.results import fold_matrix, load_results, render_markdown
from frontier_bridge.runner import (
    RunError,
    build_launch,
    launch_and_wait,
    probe_endpoint,
    verify_artifact,
)
from frontier_bridge.validation import validate_instance, validate_path

app = typer.Typer(
    name="frontier",
    help="Frontier Bridge: profile hardware, plan model execution, validate results.",
    no_args_is_help=True,
)
catalog_app = typer.Typer(help="List committed hardware and model profiles.", no_args_is_help=True)
app.add_typer(catalog_app, name="catalog")
results_app = typer.Typer(help="Fold benchmark results into reports.", no_args_is_help=True)
app.add_typer(results_app, name="results")


@app.callback(invoke_without_command=True)
def _main(
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        typer.echo(f"frontier-bridge {__version__}")
        raise typer.Exit()


@app.command()
def detect(
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write hwprofile/v1 YAML here instead of stdout."
    ),
    skip_disk_bench: bool = typer.Option(
        False, "--skip-disk-bench", help="Skip the bounded SSD read microbenchmark."
    ),
) -> None:
    """Profile this machine and emit an hwprofile/v1 document.

    Unknown values are emitted as null/unknown, never guessed.
    """
    profile = detect_hardware(run_disk_bench=not skip_disk_bench)
    text = yaml.safe_dump(profile, sort_keys=False, default_flow_style=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Wrote {output}")
    else:
        typer.echo(text)


@app.command()
def validate(
    path: Path = typer.Argument(Path("."), help="File or directory to validate."),
) -> None:
    """Validate all YAML/JSON instances declaring a schema_version against the v1 schemas."""
    report = validate_path(path.resolve())
    for issue in report.issues:
        typer.secho(f"FAIL {issue.path}: {issue.message}", fg=typer.colors.RED)
    typer.echo(
        f"{len(report.checked)} file(s) checked, {len(report.skipped)} skipped, "
        f"{len(report.issues)} issue(s)."
    )
    if not report.ok:
        raise typer.Exit(code=1)


@catalog_app.command("hardware")
def catalog_hardware() -> None:
    """List committed hardware profiles."""
    root = find_repo_root()
    profiles = list_hardware_profiles(root)
    if not profiles:
        typer.echo("No hardware profiles found.")
        raise typer.Exit()
    for p in profiles:
        typer.echo(f"{p.profile_id:32s}  method={p.method:12s}  {p.summary}")


@catalog_app.command("models")
def catalog_models() -> None:
    """List committed model profiles."""
    root = find_repo_root()
    profiles = list_model_profiles(root)
    if not profiles:
        typer.echo("No model profiles found.")
        raise typer.Exit()
    for p in profiles:
        typer.echo(f"{p.model_id:24s}  quant={p.quant:12s}  {p.summary}")


def _artifact_shard_urls(artifact: dict) -> list[str]:
    """Build resolve URLs for an artifact's shards from its HF tree source."""
    source = artifact.get("source") or ""
    shards = artifact.get("shards") or []
    if "/tree/" in source and shards:
        base, _, ref_and_dir = source.partition("/tree/")
        ref = ref_and_dir.split("/", 1)[0]
        return [f"{base}/resolve/{ref}/{s['path']}" for s in shards if s.get("path")]
    if source:
        return [source]
    return []


@catalog_app.command("inspect-gguf")
def catalog_inspect_gguf(
    locations: Optional[list[str]] = typer.Argument(
        None, help="GGUF file paths or URLs (headers only are fetched)."
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Model id from the catalog."),
    quant: Optional[str] = typer.Option(None, "--quant", help="Quant of the model artifact."),
    write: bool = typer.Option(
        False, "--write", help="Write measured sizes into the model profile's memory_model."
    ),
) -> None:
    """Measure dense vs routed-expert sizes from GGUF headers (no full download).

    Sizes come from tensor offset deltas in the header — measured, not estimated
    from quant-type tables.
    """
    root = find_repo_root()
    profile_entry = None
    if model:
        for entry in get_model_profiles(root, model):
            artifacts = entry.data.get("artifacts", [])
            match = next(
                (a for a in artifacts if quant is None or a.get("quant") == quant), None
            )
            if match is not None:
                profile_entry = entry
                locations = _artifact_shard_urls(match)
                quant = match.get("quant")
                break
        if not locations:
            typer.secho(f"error: no artifact found for {model} quant={quant}", fg=typer.colors.RED)
            raise typer.Exit(code=2)
    if not locations:
        typer.secho("error: provide GGUF locations or --model/--quant", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    typer.echo(f"Inspecting {len(locations)} shard header(s)...")
    try:
        summary = inspect_artifact(list(locations))
    except (GGUFError, OSError) as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo(yaml.safe_dump(summary, sort_keys=False))

    if write:
        if profile_entry is None or quant is None:
            typer.secho("error: --write requires --model/--quant", fg=typer.colors.RED)
            raise typer.Exit(code=2)
        data = profile_entry.data
        memory_model = data.setdefault("memory_model", {})
        memory_model.setdefault("dense_resident_gb", {})[quant] = summary["dense_resident_gb"]
        memory_model.setdefault("per_expert_gb", {})[quant] = summary["per_expert_layer_gb"]
        memory_model["measurement"] = {
            "method": summary["method"],
            "routed_experts_gb": {
                **(memory_model.get("measurement", {}).get("routed_experts_gb", {})),
                quant: summary["routed_experts_gb"],
            },
        }
        profile_entry.path.write_text(
            yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        typer.echo(f"Wrote measured memory_model into {profile_entry.path}")


@app.command()
def plan(
    model: str = typer.Argument(..., help="Model id, e.g. glm-5.2"),
    hardware: str = typer.Option(..., "--hardware", help="Hardware profile id, e.g. m5_max_128gb"),
    workload: str = typer.Option("chat", "--workload", help="Workload profile, e.g. coding_agent"),
    ctx: int = typer.Option(32768, "--ctx", help="Context budget in tokens."),
    quant: Optional[str] = typer.Option(
        None, "--quant", help="Force a specific quant; default picks the best fitting one."
    ),
    engine: Optional[str] = typer.Option(
        None, "--engine", help="Force a runtime engine (must be in the model's claimed runtimes)."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write plan/v1 YAML here instead of stdout."
    ),
) -> None:
    """Generate a rules-based execution plan (plan/v1) for a model on a hardware profile.

    Refuses gracefully with verdict: not_recommended when the numbers don't work.
    """
    root = find_repo_root()
    try:
        result = generate_plan(
            repo_root=root,
            model_id=model,
            hardware_id=hardware,
            workload=workload,
            context_budget=ctx,
            quant=quant,
            engine_override=engine,
        )
    except PlanError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    text = yaml.safe_dump(result, sort_keys=False, default_flow_style=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Wrote {output}")
    else:
        typer.echo(text)
    if result.get("verdict") == "not_recommended":
        typer.secho("verdict: not_recommended (see reasons above)", fg=typer.colors.YELLOW)


@results_app.command("matrix")
def results_matrix(
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write the matrix markdown here (e.g. docs/compatibility_matrix.md)."
    ),
) -> None:
    """Fold all committed benchresult files into the compatibility matrix."""
    root = find_repo_root()
    rows = fold_matrix(load_results(root))
    markdown = render_markdown(rows)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        header = "# Compatibility Matrix\n\nGenerated by `frontier results matrix` — do not edit by hand.\n\n"
        output.write_text(header + markdown, encoding="utf-8")
        typer.echo(f"Wrote {output} ({len(rows)} row(s))")
    else:
        typer.echo(markdown)


@app.command()
def run(
    plan_file: Path = typer.Argument(..., help="A plan/v1 YAML file (from `frontier plan -o`)."),
    model_path: Path = typer.Option(..., "--model-path", help="Path to the GGUF artifact (first shard for multi-part models)."),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Verify the artifact sha256 against the model profile pins before launching.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the exact launch command and exit without running."
    ),
    ready_timeout: float = typer.Option(
        600.0, "--ready-timeout", help="Seconds to wait for the endpoint before giving up."
    ),
) -> None:
    """Execute a plan: verify the artifact, launch the runtime, health-check the endpoint.

    The runtime keeps running in the foreground; stop it with Ctrl-C.
    """
    plan_data = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
    if plan_data.get("schema_version") != "plan/v1":
        typer.secho("error: not a plan/v1 file", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    if plan_data.get("verdict") == "not_recommended":
        typer.secho(
            "This plan is a refusal (verdict: not_recommended). Refusing to run it.",
            fg=typer.colors.YELLOW,
        )
        for reason in plan_data.get("reasons", []):
            typer.echo(f"  - {reason}")
        raise typer.Exit(code=1)

    try:
        if verify:
            if not model_path.exists():
                raise RunError(f"artifact not found: {model_path}")
            model_id, _, quant = (plan_data["inputs"]["modelprofile"]).partition("/")
            root = find_repo_root()
            entries = get_model_profiles(root, model_id)
            profile = next(
                (
                    e.data
                    for e in entries
                    if any(a.get("quant") == quant for a in e.data.get("artifacts", []))
                ),
                None,
            )
            if profile is None:
                raise RunError(f"no model profile found for {model_id}/{quant}")
            typer.echo("Verifying artifact sha256 (large files take a while)...")
            typer.echo(verify_artifact(model_path, profile, quant))

        spec = build_launch(plan_data, str(model_path))
        if dry_run:
            typer.echo(spec.command)
            raise typer.Exit()
        process = launch_and_wait(spec, echo=typer.echo, ready_timeout_s=ready_timeout)
    except RunError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(
        f"OpenAI-compatible endpoint live at http://127.0.0.1:{spec.port}/v1 "
        f"(engine: {spec.engine}). Ctrl-C to stop."
    )
    try:
        process.wait()
    except KeyboardInterrupt:
        typer.echo("Stopping runtime...")
        process.terminate()
        process.wait(timeout=30)


@app.command()
def bench(
    plan_file: Path = typer.Option(..., "--plan", help="The plan/v1 YAML this run executes."),
    suite: str = typer.Option(
        "chat", "--suite", help=f"Bundled suite ({', '.join(SUITES)}) or a JSONL path."
    ),
    port: int = typer.Option(8080, "--port", help="Port of the already-running endpoint."),
    result_id: Optional[str] = typer.Option(None, "--result-id", help="Defaults to plan_id + suite + timestamp."),
    runtime_commit: Optional[str] = typer.Option(
        None, "--runtime-commit", help="Commit/build id of the runtime under test (pin)."
    ),
    runtime_pid: Optional[int] = typer.Option(
        None, "--runtime-pid", help="Runtime process id, for peak-RSS collection."
    ),
    repro_of: Optional[list[str]] = typer.Option(
        None, "--repro-of", help="result_id(s) this run reproduces. Two are needed for verified."
    ),
    output_dir: Path = typer.Option(
        Path("results/local"), "--output-dir", help="Where the benchresult JSON lands."
    ),
) -> None:
    """Benchmark a running endpoint against a prompt suite → benchresult/v1 JSON.

    Start the runtime first (e.g. `frontier run <plan> --model-path ...`), then
    point bench at its port. Status is 'claimed' unless all four pins are set
    and two reproductions are listed — the tool enforces the verified rule.
    """
    plan_data = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
    if plan_data.get("schema_version") != "plan/v1":
        typer.secho("error: --plan must be a plan/v1 file", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    endpoint_info = probe_endpoint(port)
    if endpoint_info is None:
        typer.secho(
            f"error: no OpenAI-compatible endpoint answering on port {port}. "
            "Start the runtime first (frontier run ...).",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    served_models = [m.get("id") for m in endpoint_info.get("data", []) if m.get("id")]
    served_model = served_models[0] if served_models else None

    prompts = load_suite(suite)
    typer.echo(
        f"Running suite '{suite}' ({len(prompts)} prompts) against port {port}"
        + (f" (model: {served_model})" if served_model else "")
        + "..."
    )
    samplers, disk = start_collectors(runtime_pid)
    try:
        timings, tasks = run_suite(port, suite, prompts, echo=typer.echo, model=served_model)
    finally:
        telemetry = stop_and_report(samplers, disk)

    # Pins: model sha256 comes from the profile pin for the plan's quant.
    model_id, _, quant = (plan_data.get("inputs", {}).get("modelprofile", "")).partition("/")
    model_sha = None
    for entry in get_model_profiles(find_repo_root(), model_id):
        for artifact in entry.data.get("artifacts", []):
            if artifact.get("quant") == quant and artifact.get("sha256"):
                model_sha = artifact["sha256"]
                break

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rid = result_id or f"{plan_data.get('plan_id', 'plan')}-{suite}-{stamp}"
    result = build_result(
        result_id=rid,
        suite_name=suite,
        timings=timings,
        tasks=tasks,
        telemetry=telemetry,
        pins={
            "plan_hash": plan_hash(plan_data),
            "model_sha256": model_sha,
            "runtime_commit": runtime_commit,
            "hwprofile_id": plan_data.get("inputs", {}).get("hwprofile"),
        },
        context_len=plan_data.get("inputs", {}).get("context_budget"),
        reproductions=list(repro_of or []),
        subject={
            "modelprofile": plan_data.get("inputs", {}).get("modelprofile"),
            "workload": suite,
        },
    )
    errors = validate_instance(result)
    if errors:
        typer.secho(f"internal error: result failed schema validation: {errors}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{rid}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    passed = sum(1 for t in tasks if t["passed"])
    typer.echo(
        f"\n{passed}/{len(tasks)} tasks passed | ttft={result['metrics']['ttft_ms']}ms "
        f"decode={result['metrics']['decode_tps']}tps "
        f"p95={result['metrics']['token_latency_ms']['p95']}ms | status={result['status']}"
    )
    typer.echo(f"Wrote {out_path}")
