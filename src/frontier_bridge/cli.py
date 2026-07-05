"""The `frontier` command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml

from frontier_bridge import __version__
from frontier_bridge.catalog import find_repo_root, list_hardware_profiles, list_model_profiles
from frontier_bridge.detect import detect_hardware
from frontier_bridge.planner.engine import PlanError, generate_plan
from frontier_bridge.validation import validate_path

app = typer.Typer(
    name="frontier",
    help="Frontier Bridge: profile hardware, plan model execution, validate results.",
    no_args_is_help=True,
)
catalog_app = typer.Typer(help="List committed hardware and model profiles.", no_args_is_help=True)
app.add_typer(catalog_app, name="catalog")


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


@app.command()
def plan(
    model: str = typer.Argument(..., help="Model id, e.g. glm-5.2"),
    hardware: str = typer.Option(..., "--hardware", help="Hardware profile id, e.g. m5_max_128gb"),
    workload: str = typer.Option("chat", "--workload", help="Workload profile, e.g. coding_agent"),
    ctx: int = typer.Option(32768, "--ctx", help="Context budget in tokens."),
    quant: Optional[str] = typer.Option(
        None, "--quant", help="Force a specific quant; default picks the best fitting one."
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
