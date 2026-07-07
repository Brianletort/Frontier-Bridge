"""The `frontier` command-line interface."""

from __future__ import annotations

import json
import time
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
from frontier_bridge.bench.experiments import (
    append_jsonl,
    kv_per_1k_tokens_mb,
    ladder_rung,
    sweep_point,
)
from frontier_bridge.bench.streamspike import expert_read_bench, stream_read_gbps
from frontier_bridge.detect import detect_hardware
from frontier_bridge.fleet import (
    fleet_plan as run_fleet_plan,
    load_fleet,
    remote_bench,
    remote_detect,
)
from frontier_bridge.gguf import GGUFError, inspect_artifact
from frontier_bridge.ingest import IngestError, ingest_repo
from frontier_bridge.planner.engine import PlanError, generate_plan
from frontier_bridge.provision import (
    SetupError,
    download_artifact,
    ensure_runtime,
    install_hint,
)
from frontier_bridge.results import fold_matrix, load_results, render_markdown
from frontier_bridge.runbook import (
    RunbookEntry,
    load_runbooks,
    match_runbook,
    render_markdown as render_runbook_markdown,
    verify_runbook,
)
from frontier_bridge.ui import serve as ui_serve
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
bench_app = typer.Typer(
    help="Benchmark a running endpoint, or run experiment harnesses.",
    invoke_without_command=True,
)
app.add_typer(bench_app, name="bench")
runbook_app = typer.Typer(
    help="Hardware-class runbooks: match this machine, render, verify provenance (RFC 0003).",
    no_args_is_help=True,
)
app.add_typer(runbook_app, name="runbook")
fleet_app = typer.Typer(
    help="Operate a fleet of machines: plan across profiles, run detect/bench remotely (RFC 0005).",
    no_args_is_help=True,
)
app.add_typer(fleet_app, name="fleet")


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


@catalog_app.command("add")
def catalog_add(
    repo: str = typer.Argument(..., help="HF repo (org/name) or URL, optionally /tree/<ref>."),
    quant: Optional[str] = typer.Option(
        None, "--quant", help="Quant variant to ingest (dir name or quant token)."
    ),
    model_id: Optional[str] = typer.Option(
        None, "--model-id", help="Catalog model id; derived from the repo name by default."
    ),
    family: Optional[str] = typer.Option(None, "--family", help="Model family label."),
    inspect: bool = typer.Option(
        True,
        "--inspect/--no-inspect",
        help="Range-inspect GGUF headers to measure the memory model (recommended).",
    ),
    write: bool = typer.Option(
        False, "--write", help="Write the profile into model_profiles/ (else print it)."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing profile file."),
) -> None:
    """Generate a modelprofile/v1 draft from a Hugging Face GGUF repo.

    sha256 pins come from the repo's LFS metadata; sizes come from the file
    listing; architecture and memory-model figures are measured from GGUF
    headers via range requests (no full download). Values that cannot be
    measured stay null — review the draft before committing it.
    """
    try:
        profile = ingest_repo(
            repo, model_id=model_id, quant=quant, family=family, inspect_headers=inspect
        )
    except (IngestError, GGUFError, OSError) as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    errors = validate_instance(profile)
    if errors:
        typer.secho(f"internal error: generated profile fails schema: {errors}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    text = yaml.safe_dump(profile, sort_keys=False, default_flow_style=False)
    if not write:
        typer.echo(text)
        typer.echo("# Draft only — rerun with --write to save into model_profiles/.")
        return
    root = find_repo_root()
    model_dir = profile["model_id"].replace(".", "_").replace("-", "_")
    out_path = root / "model_profiles" / model_dir / f"{profile['artifacts'][0]['quant']}.yaml"
    if out_path.exists() and not force:
        typer.secho(f"error: {out_path} exists (use --force to overwrite)", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    typer.echo(f"Wrote {out_path}")
    if profile["architecture"]["params_total_b"] is None:
        typer.secho(
            "note: params_total_b is null (headers not inspected) — the planner "
            "will refuse this model until it is measured.",
            fg=typer.colors.YELLOW,
        )


@catalog_app.command("kv-from-ladder")
def catalog_kv_from_ladder(
    ladder_file: Path = typer.Argument(..., help="Context-ladder JSONL (frontier bench context-ladder)."),
    model: str = typer.Option(..., "--model", help="Model id whose profiles get the measurement."),
    quant: Optional[str] = typer.Option(
        None, "--quant", help="Write only into this quant's profile (default: all of the model's)."
    ),
) -> None:
    """Fold context-ladder measurements into the model's kv_per_1k_tokens_mb.

    Only stable rungs with a parsed KV size count; the f16 (unquantized KV)
    figure lands in memory_model.kv_per_1k_tokens_mb, and the full per-kv-quant
    fold is recorded under memory_model.measurement.
    """
    records = [
        json.loads(line)
        for line in ladder_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    folded = kv_per_1k_tokens_mb(records)
    if not folded:
        typer.secho(
            "error: no stable rungs with parsed KV sizes in this ladder file",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    typer.echo(f"Measured KV MB per 1K tokens: {folded}")

    root = find_repo_root()
    entries = get_model_profiles(root, model)
    if not entries:
        typer.secho(f"error: model {model!r} not found", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    written = 0
    for entry in entries:
        artifacts = entry.data.get("artifacts", [])
        if quant is not None and not any(a.get("quant") == quant for a in artifacts):
            continue
        memory_model = entry.data.setdefault("memory_model", {})
        if "f16" in folded:
            memory_model["kv_per_1k_tokens_mb"] = folded["f16"]
        measurement = memory_model.setdefault("measurement", {})
        measurement["kv_per_1k_tokens_mb_by_kv_quant"] = folded
        measurement["kv_method"] = "context_ladder_server_log_kv_size"
        entry.path.write_text(
            yaml.safe_dump(entry.data, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        typer.echo(f"Wrote {entry.path}")
        written += 1
    if written == 0:
        typer.secho("error: no matching profile files", fg=typer.colors.RED)
        raise typer.Exit(code=2)


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


@runbook_app.command("match")
def runbook_match(
    profile: Optional[Path] = typer.Option(
        None,
        "--profile",
        help="An hwprofile/v1 YAML to match against (default: run detect on this machine).",
    ),
    show_unmet: bool = typer.Option(
        False, "--show-unmet", help="Explain why non-matching runbooks did not match."
    ),
) -> None:
    """Find the runbooks that apply to a machine.

    With no --profile, this machine is detected first (disk bench skipped for speed).
    """
    root = find_repo_root()
    runbooks = load_runbooks(root)
    if not runbooks:
        typer.echo("No runbooks committed yet.")
        raise typer.Exit()
    if profile is not None:
        hwprofile = yaml.safe_load(profile.read_text(encoding="utf-8"))
    else:
        typer.echo("Profiling this machine (disk bench skipped)...")
        hwprofile = detect_hardware(run_disk_bench=False)
    matched_any = False
    for entry in runbooks:
        report = match_runbook(entry.data, hwprofile)
        if report.matched:
            matched_any = True
            typer.secho(f"MATCH {entry.runbook_id}", fg=typer.colors.GREEN)
            typer.echo(f"      {entry.data.get('title', '')}")
        elif show_unmet:
            typer.echo(f"no    {entry.runbook_id}")
            for reason in report.unmet:
                typer.echo(f"      unmet: {reason}")
    if not matched_any:
        typer.echo(
            "No runbook matches this machine yet. `frontier plan` still works "
            "against your detected profile — and a runbook for your hardware "
            "class would be a welcome contribution."
        )


@runbook_app.command("render")
def runbook_render(
    runbook_file: Optional[Path] = typer.Argument(
        None, help="A runbook/v1 YAML file (default: render all committed runbooks)."
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Write rendered markdown here (default: runbooks/rendered/).",
    ),
) -> None:
    """Render runbook YAML to distributable markdown.

    The YAML is the source of truth; rendered files are build products,
    regenerated by CI and never hand-edited.
    """
    root = find_repo_root()
    if runbook_file is not None:
        entries = [
            e
            for e in load_runbooks(root)
            if e.path.resolve() == runbook_file.resolve()
        ]
        if not entries:
            data = yaml.safe_load(runbook_file.read_text(encoding="utf-8"))
            entries = [RunbookEntry(path=runbook_file, data=data)]
    else:
        entries = load_runbooks(root)
    if not entries:
        typer.echo("No runbooks to render.")
        raise typer.Exit()
    dest = output_dir or (root / "runbooks" / "rendered")
    dest.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        out_path = dest / f"{entry.runbook_id}.md"
        out_path.write_text(render_runbook_markdown(entry.data), encoding="utf-8")
        typer.echo(f"Wrote {out_path}")


@runbook_app.command("verify")
def runbook_verify(
    path: Optional[Path] = typer.Argument(
        None, help="A runbook file or directory (default: runbooks/)."
    ),
) -> None:
    """CI provenance gate: every expected number must trace to a committed
    benchresult, and every menu verdict must agree with its committed plan."""
    root = find_repo_root()
    if path is None or path.is_dir():
        entries = load_runbooks(root)
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        entries = [RunbookEntry(path=path, data=data)]
    if not entries:
        typer.echo("No runbooks to verify.")
        raise typer.Exit()
    failed = False
    for entry in entries:
        errors = verify_runbook(entry.data, root)
        if errors:
            failed = True
            for message in errors:
                typer.secho(f"FAIL {entry.path}: {message}", fg=typer.colors.RED)
        else:
            typer.echo(f"ok   {entry.path}")
    if failed:
        raise typer.Exit(code=1)


@fleet_app.command("plan")
def fleet_plan_cmd(
    model: str = typer.Argument(..., help="Model id, e.g. deepseek-v4-flash"),
    workload: str = typer.Option("chat", "--workload", help="Workload profile."),
    ctx: int = typer.Option(32768, "--ctx", help="Context budget in tokens."),
    quant: Optional[str] = typer.Option(None, "--quant", help="Force a specific quant."),
    fleet_file: Optional[Path] = typer.Option(
        None, "--fleet", help="fleet/v1 YAML (default: first found in fleet/local/ then fleet/)."
    ),
) -> None:
    """Which of my machines should run this model? Runs the existing planner
    against every registered machine's committed profile — ranked by verdict
    class only, no new scoring math."""
    root = find_repo_root()
    machines = load_fleet(root, fleet_file)
    if not machines:
        typer.secho(
            "No fleet registry found. Create fleet/local/<name>.yaml (fleet/v1); "
            "see fleet/example.yaml.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=2)
    rows = run_fleet_plan(root, machines, model, workload, ctx, quant=quant)
    typer.echo(f"{'Machine':<16} {'Profile':<34} {'Verdict':<16} Quant / reasons")
    for row in rows:
        detail = row.quant or "; ".join(row.reasons) or ""
        color = {
            "recommended": typer.colors.GREEN,
            "experimental": typer.colors.YELLOW,
        }.get(row.verdict, typer.colors.RED)
        typer.secho(
            f"{row.machine:<16} {row.hwprofile or '—':<34} {row.verdict:<16} {detail}",
            fg=color,
        )


@fleet_app.command("detect")
def fleet_detect_cmd(
    machine_name: str = typer.Argument(..., help="Machine name from the fleet registry."),
    fleet_file: Optional[Path] = typer.Option(None, "--fleet", help="fleet/v1 YAML."),
    skip_disk_bench: bool = typer.Option(
        False, "--skip-disk-bench", help="Skip the SSD microbenchmark on the remote."
    ),
) -> None:
    """Run `frontier detect` on a remote machine over SSH and pull the profile
    back into hardware_profiles/local/ for review before commit."""
    root = find_repo_root()
    machines = {m.name: m for m in load_fleet(root, fleet_file)}
    if machine_name not in machines:
        typer.secho(
            f"error: {machine_name!r} is not in the fleet registry "
            f"(known: {', '.join(sorted(machines)) or 'none'})",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    path = remote_detect(
        machines[machine_name], root, echo=typer.echo, skip_disk_bench=skip_disk_bench
    )
    if path is None:
        raise typer.Exit(code=1)
    typer.echo(f"Profile pulled to {path} — review it (no hostnames/serials), then commit.")


@fleet_app.command(
    "bench",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def fleet_bench_cmd(
    ctx: typer.Context,
    machine_name: str = typer.Argument(..., help="Machine name from the fleet registry."),
    fleet_file: Optional[Path] = typer.Option(None, "--fleet", help="fleet/v1 YAML."),
) -> None:
    """Run `frontier bench <args>` on a remote machine and pull results/local back.

    Everything after the machine name is passed to the remote bench verbatim.
    """
    root = find_repo_root()
    machines = {m.name: m for m in load_fleet(root, fleet_file)}
    if machine_name not in machines:
        typer.secho(f"error: {machine_name!r} is not in the fleet registry", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    result_dir = remote_bench(machines[machine_name], root, list(ctx.args), echo=typer.echo)
    if result_dir is None:
        raise typer.Exit(code=1)
    typer.echo(f"Results pulled to {result_dir} — review, then move keepers into results/community/ or verified/.")


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
def ui(
    port: int = typer.Option(7861, "--port", help="Port for the local web UI."),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the UI in the default browser."
    ),
) -> None:
    """Local web UI: hardware view, model catalog, side-by-side plan comparison,
    one-click setup/run, and the verified results matrix.

    Serves on 127.0.0.1 only. The UI calls the same library code as the CLI.
    """
    ui_serve(port=port, open_browser=open_browser)


@app.command()
def setup(
    plan_file: Path = typer.Argument(..., help="A plan/v1 YAML file (from `frontier plan -o`)."),
    dest: Path = typer.Option(
        Path("models"), "--dest", help="Directory for downloaded artifacts."
    ),
    launch: bool = typer.Option(
        True, "--launch/--no-launch", help="Start the runtime once the artifact is ready."
    ),
    auto_install: bool = typer.Option(
        True,
        "--auto-install/--no-auto-install",
        help="Install llama.cpp via Homebrew when it is missing (other runtimes print their route).",
    ),
    verify: bool = typer.Option(
        True, "--verify/--no-verify", help="Verify each shard's sha256 against the profile pin."
    ),
    allow_unpinned: bool = typer.Option(
        False, "--allow-unpinned", help="Download shards that have no sha256 pin (discouraged)."
    ),
    port: Optional[int] = typer.Option(None, "--port", help="Override the plan's port at launch."),
    ready_timeout: float = typer.Option(
        600.0, "--ready-timeout", help="Seconds to wait for the endpoint at launch."
    ),
) -> None:
    """One command from plan to running endpoint: install the runtime, download
    the pinned artifact (resumable, hash-verified, space-checked), and launch.

    Orchestration only — `frontier run` does the launching underneath.
    """
    plan_data = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
    if plan_data.get("schema_version") != "plan/v1":
        typer.secho("error: not a plan/v1 file", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    if plan_data.get("verdict") == "not_recommended":
        typer.secho(
            "This plan is a refusal (verdict: not_recommended). Refusing to set it up.",
            fg=typer.colors.YELLOW,
        )
        for reason in plan_data.get("reasons", []):
            typer.echo(f"  - {reason}")
        raise typer.Exit(code=1)

    engine = (plan_data.get("runtime") or {}).get("engine") or "?"
    model_id, _, quant = (plan_data.get("inputs", {}).get("modelprofile", "")).partition("/")

    # 1. Runtime.
    typer.echo(f"[1/3] Runtime: {engine}")
    if ensure_runtime(engine, echo=typer.echo, auto_install=auto_install):
        typer.echo(f"  {engine} is available.")
    else:
        typer.secho(
            f"  {engine} is not available; install it and rerun. "
            f"({install_hint(engine).splitlines()[0]})",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    # 2. Artifact.
    typer.echo(f"[2/3] Artifact: {model_id}/{quant} -> {dest}")
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
        typer.secho(f"error: no model profile found for {model_id}/{quant}", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    try:
        model_path = download_artifact(
            profile,
            quant,
            dest,
            echo=typer.echo,
            verify=verify,
            allow_unpinned=allow_unpinned,
        )
    except SetupError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo(f"  Artifact ready: {model_path}")

    # 3. Launch.
    if not launch:
        typer.echo(
            f"[3/3] Skipped launch (--no-launch). Run it later with:\n"
            f"  frontier run {plan_file} --model-path {model_path}"
        )
        return
    typer.echo("[3/3] Launching...")
    try:
        # Shards were verified at download time; no need to re-hash here.
        spec = build_launch(plan_data, str(model_path), port_override=port)
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
    port: Optional[int] = typer.Option(
        None, "--port", help="Override the port in the plan's launch command."
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

        spec = build_launch(plan_data, str(model_path), port_override=port)
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


@bench_app.callback()
def bench(
    ctx: typer.Context,
    plan_file: Optional[Path] = typer.Option(
        None, "--plan", help="The plan/v1 YAML this run executes."
    ),
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

    Subcommands (sweep-offload, context-ladder) are experiment harnesses that
    launch llama-server themselves.
    """
    if ctx.invoked_subcommand is not None:
        return
    if plan_file is None:
        typer.secho("error: --plan is required (or use a subcommand)", fg=typer.colors.RED)
        raise typer.Exit(code=2)
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


@bench_app.command("sweep-offload")
def bench_sweep_offload(
    model: Path = typer.Option(..., "--model", help="GGUF path (first shard for multi-part)."),
    values: str = typer.Option(..., "--values", help="Comma-separated --n-cpu-moe values to sweep."),
    ctx_tokens: int = typer.Option(8192, "--ctx", help="Context size for every point."),
    port: int = typer.Option(8899, "--port", help="Port llama-server binds per point."),
    output: Path = typer.Option(..., "--output", help="JSONL output (results/experiments/...)."),
    extra: str = typer.Option("", "--extra", help="Extra llama-server args, space-separated."),
) -> None:
    """Sweep llama.cpp --n-cpu-moe values to find the working-set stability boundary.

    Each point: launch llama-server, health-check, short decode probes, teardown,
    one JSONL record. This is an experiment harness, not a benchmark — probes are
    short on purpose; interesting points get full `frontier bench` runs afterwards.
    """
    extra_args = extra.split() if extra else []
    for value in [int(v) for v in values.split(",")]:
        typer.echo(f"--- n-cpu-moe={value}")
        record = sweep_point(str(model), value, ctx_tokens, port, extra_args)
        typer.echo(json.dumps(record))
        append_jsonl(output, record)
        time.sleep(5)  # let memory settle between points
    typer.echo(f"Wrote {output}")


@bench_app.command("ssd-stream")
def bench_ssd_stream(
    chunks: str = typer.Option(
        "4,16,64", "--chunks", help="Comma-separated read sizes in MB (expert-slice scale)."
    ),
    file_gb: float = typer.Option(4.0, "--file-gb", help="Scratch file size."),
    reads: int = typer.Option(48, "--reads", help="Random reads per chunk size."),
    directory: Optional[Path] = typer.Option(
        None, "--dir", help="Directory on the SSD under test (default: tmp)."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Append records to this JSONL (results/experiments/...)."
    ),
) -> None:
    """Measure uncached random-read bandwidth at expert-slice granularity.

    This is the measured floor of L2 stream-on-miss: sequential bandwidth
    flatters the SSD, but expert misses read megabytes at random offsets.
    Record the worst figure as `measured.expert_read_gbps` on the hardware
    profile's storage node so plans use it in their streaming math.
    """
    chunk_list = [int(c) for c in chunks.split(",")]
    typer.echo(
        f"Uncached random reads over a {file_gb}GB scratch file "
        f"({reads} reads per size)..."
    )
    records = expert_read_bench(
        chunk_list, file_gb=file_gb, reads_per_size=reads,
        directory=str(directory) if directory else None,
    )
    for record in records:
        typer.echo(
            f"  {record['chunk_mb']:>4} MB chunks: {record['gbps']} GB/s "
            f"({record['reads_per_s']} reads/s)"
        )
        if output:
            append_jsonl(output, record)
    floor = stream_read_gbps(records)
    typer.echo(
        f"Planner figure (worst across sizes): expert_read_gbps = {floor}"
    )
    if output:
        typer.echo(f"Wrote {output}")


@bench_app.command("context-ladder")
def bench_context_ladder(
    model: Path = typer.Option(..., "--model", help="GGUF path (first shard for multi-part)."),
    rungs: str = typer.Option(..., "--rungs", help="Comma-separated context sizes."),
    n_cpu_moe: int = typer.Option(..., "--n-cpu-moe", help="Offload value for every rung."),
    port: int = typer.Option(8899, "--port", help="Port llama-server binds per rung."),
    kv_quant: Optional[str] = typer.Option(
        None, "--kv-quant", help="KV cache quant for -ctk/-ctv, e.g. q8_0."
    ),
    output: Path = typer.Option(..., "--output", help="JSONL output (results/experiments/...)."),
) -> None:
    """Context ladder: measure KV footprint, long-context TTFT/decode, and needle
    retrieval at increasing context sizes.

    Each rung: server start → health → KV size from the init log → needle probe
    at ~70% of ctx → teardown → one JSONL record.
    """
    for rung_ctx in [int(v) for v in rungs.split(",")]:
        typer.echo(f"--- ctx={rung_ctx} kv={kv_quant or 'f16'}")
        record = ladder_rung(str(model), rung_ctx, n_cpu_moe, port, kv_quant)
        typer.echo(json.dumps(record))
        append_jsonl(output, record)
        time.sleep(5)
    typer.echo(f"Wrote {output}")
