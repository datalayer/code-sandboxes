# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Typer CLI for code sandboxes: REPL sessions and CRUD management."""

from __future__ import annotations

import os
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import Sandbox
from .manage import SandboxManagementError, get_manager, manageable_variants
from .models import Result

app = typer.Typer(help="Code sandboxes: run a REPL, list, create and delete sandboxes.")

console = Console()

_SUPPORTED_REPL_VARIANTS = {
    "jupyter",
    "docker",
    "eval",
    "monty",
    "google_colab",
    "google-colab",
    "kaggle",
    "modal",
    "datalayer",
}

_EXIT_COMMANDS = {":exit", ":quit", "exit", "quit"}


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """Code sandboxes CLI."""
    if ctx.invoked_subcommand is None:
        _run_repl(variant="jupyter")


def _print_result(result: Any) -> None:
    if not getattr(result, "execution_ok", True):
        msg = getattr(result, "execution_error", None) or "Execution failed"
        typer.secho(msg, fg=typer.colors.RED)
        return

    for line in getattr(result, "stdout", "").splitlines():
        typer.echo(line)

    for line in getattr(result, "stderr", "").splitlines():
        typer.secho(line, fg=typer.colors.YELLOW)

    code_error = getattr(result, "code_error", None)
    if code_error is not None:
        typer.secho(f"{code_error.name}: {code_error.value}", fg=typer.colors.RED)

    # Prefer the main result and avoid duplicating stdout.
    text = None
    results = getattr(result, "results", [])
    for item in results:
        if isinstance(item, Result) and item.is_main_result:
            text = item.text
            break
    if text:
        typer.echo(text)


def _resolve_variant(variant: str | None) -> str:
    if variant:
        selected = variant.strip().lower()
    else:
        selected = typer.prompt(
            "Sandbox variant",
            default="jupyter",
            show_default=True,
        )
        selected = selected.strip().lower()

    if selected not in _SUPPORTED_REPL_VARIANTS:
        raise typer.BadParameter(
            f"Unsupported variant: {selected}. Supported values: "
            + ", ".join(sorted(_SUPPORTED_REPL_VARIANTS))
        )
    if selected == "google-colab":
        return "google_colab"
    return selected


def _resolve_variant_kwargs(
    variant: str,
    server_url: str | None,
    kernel_id: str | None,
    proxy_token: str | None,
    token: str | None,
    run_url: str | None,
    gpu: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    if variant == "jupyter":
        # Match `jupyter console` behavior by launching local Jupyter on random port.
        kwargs["port"] = 0

    if variant == "google_colab":
        kwargs["server_url"] = server_url or typer.prompt("Colab runtime URL (RUNTIME_URL)")
        kwargs["kernel_id"] = kernel_id or typer.prompt("Colab kernel id (RUNTIME_ID)")
        kwargs["proxy_token"] = proxy_token or typer.prompt(
            "Colab runtime proxy token (RUNTIME_PROXY_TOKEN)",
            hide_input=True,
        )

    if variant == "kaggle":
        kwargs["server_url"] = server_url or typer.prompt("Kaggle runtime proxy URL (RUNTIME_URL)")
        # kernel_id is optional: leave empty to create a new kernel (needs a token).
        resolved_kernel_id = kernel_id or typer.prompt(
            "Kaggle kernel id (RUNTIME_ID, leave empty to create a new kernel)",
            default="",
            show_default=False,
        )
        if resolved_kernel_id:
            kwargs["kernel_id"] = resolved_kernel_id
        resolved_token = token or os.environ.get("KAGGLE_API_TOKEN")
        if resolved_token:
            kwargs["token"] = resolved_token

    if variant == "datalayer":
        if token:
            kwargs["token"] = token
        if run_url:
            kwargs["run_url"] = run_url

    if variant in {"modal", "datalayer", "kaggle"} and gpu:
        kwargs["gpu"] = gpu

    return kwargs


def _run_repl(
    variant: str | None = None,
    timeout: float = 60.0,
    environment: str | None = None,
    server_url: str | None = None,
    kernel_id: str | None = None,
    proxy_token: str | None = None,
    token: str | None = None,
    run_url: str | None = None,
    gpu: str | None = None,
) -> None:
    selected_variant = _resolve_variant(variant)
    sandbox_kwargs = _resolve_variant_kwargs(
        selected_variant,
        server_url=server_url,
        kernel_id=kernel_id,
        proxy_token=proxy_token,
        token=token,
        run_url=run_url,
        gpu=gpu,
    )

    typer.secho(f"Starting sandbox variant: {selected_variant}", fg=typer.colors.CYAN)

    try:
        with Sandbox.create(
            variant=selected_variant,
            timeout=timeout,
            environment=environment,
            **sandbox_kwargs,
        ) as sandbox:
            sandbox_id = sandbox.sandbox_id or "<unknown>"
            typer.secho(
                f"Sandbox started (id={sandbox_id}). Type code and press Enter.",
                fg=typer.colors.GREEN,
            )
            typer.echo("Use :exit or Ctrl+D to terminate.")

            while True:
                try:
                    code = input(">>> ")
                except EOFError:
                    typer.echo("")
                    break
                except KeyboardInterrupt:
                    typer.echo("\n(Interrupted. Type :exit to quit.)")
                    continue

                if not code.strip():
                    continue
                if code.strip() in _EXIT_COMMANDS:
                    break

                try:
                    result = sandbox.run_code(code)
                except KeyboardInterrupt:
                    typer.echo("\n(Execution interrupted.)")
                    continue
                except Exception as exc:
                    typer.secho(f"Execution failed: {exc}", fg=typer.colors.RED)
                    continue

                _print_result(result)
    except Exception as exc:
        typer.secho(f"Failed to start REPL: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from None

    typer.secho("Sandbox terminated.", fg=typer.colors.GREEN)


@app.command()
def repl(
    variant: str | None = typer.Option(
        None,
        "--variant",
        "-v",
        help=(
            "Sandbox variant (jupyter, docker, eval, monty, "
            "google_colab/google-colab, kaggle, modal, datalayer)."
        ),
    ),
    timeout: float = typer.Option(60.0, help="Default code execution timeout (seconds)."),
    environment: str | None = typer.Option(
        None,
        help="Sandbox environment (used by variants such as datalayer).",
    ),
    server_url: str | None = typer.Option(None, help="Colab runtime URL."),
    kernel_id: str | None = typer.Option(None, help="Colab kernel ID."),
    proxy_token: str | None = typer.Option(None, help="Colab runtime proxy token."),
    token: str | None = typer.Option(None, help="Datalayer API token override."),
    run_url: str | None = typer.Option(None, help="Datalayer run URL override."),
    gpu: str | None = typer.Option(
        None,
        "--gpu",
        help=(
            "GPU flavor / accelerator for supported variants "
            "(modal/datalayer examples: T4, A10G, A100, H100; "
            "kaggle examples: NvidiaTeslaT4, NvidiaTeslaP100, or aliases T4/P100)."
        ),
    ),
) -> None:
    """Launch an interactive REPL against the selected sandbox variant.

    The sandbox is always terminated when this command exits.
    """
    _run_repl(
        variant=variant,
        timeout=timeout,
        environment=environment,
        server_url=server_url,
        kernel_id=kernel_id,
        proxy_token=proxy_token,
        token=token,
        run_url=run_url,
        gpu=gpu,
    )


def _manager_kwargs(
    server_url: str | None = None,
    token: str | None = None,
    proxy_token: str | None = None,
    run_url: str | None = None,
    app_name: str | None = None,
    username: str | None = None,
) -> dict[str, Any]:
    """Only what was actually given: each manager keeps its own defaults."""
    kwargs = {
        "server_url": server_url,
        "token": token,
        "proxy_token": proxy_token,
        "run_url": run_url,
        "app_name": app_name,
        "username": username,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def _sandbox_table(title: str) -> Table:
    table = Table(title=title)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Variant", style="magenta")
    table.add_column("Name")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")
    return table


def _add_sandbox_row(table: Table, info: Any) -> None:
    details = ", ".join(
        f"{key}={value}"
        for key, value in (info.metadata or {}).items()
        if value not in (None, "", 0)
    )
    status = info.status.value if hasattr(info.status, "value") else str(info.status)
    table.add_row(info.id, info.variant, info.name or "", status, details)


# The options shared by the management commands, declared once.
_VARIANT_OPTION = typer.Option(
    None,
    "--variant",
    "-v",
    help="Sandbox variant (" + ", ".join(manageable_variants()) + ").",
)
_SERVER_URL_OPTION = typer.Option(None, help="Server URL (jupyter, google_colab).")
_TOKEN_OPTION = typer.Option(None, help="API token (jupyter, datalayer).")
_PROXY_TOKEN_OPTION = typer.Option(None, help="Colab runtime proxy token.")
_RUN_URL_OPTION = typer.Option(None, help="Datalayer run URL override.")
_APP_NAME_OPTION = typer.Option(None, help="Modal app name (default code-sandboxes).")
_USERNAME_OPTION = typer.Option(None, help="Kaggle username override.")
_TAG_OPTION = typer.Option(None, "--tag", help="Tag as key=value, repeatable (modal).")
_CAPABILITY_OPTION = typer.Option(
    None, "--capability", help="Capability, repeatable (datalayer)."
)


@app.command("list")
def list_sandboxes(
    variant: str | None = _VARIANT_OPTION,
    server_url: str | None = _SERVER_URL_OPTION,
    token: str | None = _TOKEN_OPTION,
    proxy_token: str | None = _PROXY_TOKEN_OPTION,
    run_url: str | None = _RUN_URL_OPTION,
    app_name: str | None = _APP_NAME_OPTION,
    username: str | None = _USERNAME_OPTION,
) -> None:
    """List sandboxes — of one variant, or of every variant that answers."""
    kwargs = _manager_kwargs(
        server_url, token, proxy_token, run_url, app_name, username
    )
    variants = [variant] if variant else manageable_variants()
    table = _sandbox_table("Code Sandboxes")
    skipped: list[tuple[str, str]] = []
    total = 0
    for name in variants:
        try:
            for info in get_manager(name, **kwargs).list():
                _add_sandbox_row(table, info)
                total += 1
        except Exception as exc:
            # A variant with no backend, no credentials, or no daemon must
            # not hide the ones that answered.
            skipped.append((name, str(exc)))
            if variant:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(code=1) from None
    if total:
        console.print(table)
    else:
        console.print("[yellow]No sandboxes found.[/yellow]")
    for name, reason in skipped:
        console.print(f"[dim]({name} skipped: {reason})[/dim]")


@app.command()
def get(
    sandbox_id: str = typer.Argument(..., help="The sandbox to look up."),
    variant: str = typer.Option(..., "--variant", "-v", help="Sandbox variant."),
    server_url: str | None = _SERVER_URL_OPTION,
    token: str | None = _TOKEN_OPTION,
    proxy_token: str | None = _PROXY_TOKEN_OPTION,
    run_url: str | None = _RUN_URL_OPTION,
    app_name: str | None = _APP_NAME_OPTION,
    username: str | None = _USERNAME_OPTION,
) -> None:
    """Show one sandbox."""
    kwargs = _manager_kwargs(
        server_url, token, proxy_token, run_url, app_name, username
    )
    try:
        info = get_manager(variant, **kwargs).get(sandbox_id)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    if info is None:
        console.print(f"[yellow]No {variant} sandbox found: {sandbox_id}[/yellow]")
        raise typer.Exit(code=1)
    table = _sandbox_table("Code Sandbox")
    _add_sandbox_row(table, info)
    console.print(table)


@app.command()
def delete(
    sandbox_id: str = typer.Argument(..., help="The sandbox to delete."),
    variant: str = typer.Option(..., "--variant", "-v", help="Sandbox variant."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation."),
    server_url: str | None = _SERVER_URL_OPTION,
    token: str | None = _TOKEN_OPTION,
    proxy_token: str | None = _PROXY_TOKEN_OPTION,
    run_url: str | None = _RUN_URL_OPTION,
    app_name: str | None = _APP_NAME_OPTION,
    username: str | None = _USERNAME_OPTION,
) -> None:
    """Delete a sandbox."""
    kwargs = _manager_kwargs(
        server_url, token, proxy_token, run_url, app_name, username
    )
    if not yes and not typer.confirm(f"Delete {variant} sandbox {sandbox_id}?"):
        raise typer.Exit(code=0)
    try:
        deleted = get_manager(variant, **kwargs).delete(sandbox_id)
    except SandboxManagementError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    if deleted:
        console.print(f"[green]Deleted {variant} sandbox {sandbox_id}.[/green]")
    else:
        console.print(f"[yellow]No {variant} sandbox found: {sandbox_id}[/yellow]")
        raise typer.Exit(code=1)


@app.command()
def update(
    sandbox_id: str = typer.Argument(..., help="The sandbox to update."),
    variant: str = typer.Option(..., "--variant", "-v", help="Sandbox variant."),
    name: str | None = typer.Option(None, help="New name (docker)."),
    tag: list[str] | None = _TAG_OPTION,
    capability: list[str] | None = _CAPABILITY_OPTION,
    code: str | None = typer.Option(
        None, help="New code, pushed as a new version (kaggle)."
    ),
    server_url: str | None = _SERVER_URL_OPTION,
    token: str | None = _TOKEN_OPTION,
    proxy_token: str | None = _PROXY_TOKEN_OPTION,
    run_url: str | None = _RUN_URL_OPTION,
    app_name: str | None = _APP_NAME_OPTION,
    username: str | None = _USERNAME_OPTION,
) -> None:
    """Update a sandbox: what changes depends on the variant."""
    manager_kwargs = _manager_kwargs(
        server_url, token, proxy_token, run_url, app_name, username
    )
    changes: dict[str, Any] = {}
    if name:
        changes["name"] = name
    if tag:
        tags: dict[str, str] = {}
        for entry in tag:
            key, separator, value = entry.partition("=")
            if not separator:
                console.print(f"[red]Not a key=value tag: {entry}[/red]")
                raise typer.Exit(code=1)
            tags[key] = value
        changes["tags"] = tags
    if capability:
        changes["capabilities"] = list(capability)
    if code is not None:
        changes["code"] = code
    try:
        info = get_manager(variant, **manager_kwargs).update(sandbox_id, **changes)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    table = _sandbox_table("Updated")
    _add_sandbox_row(table, info)
    console.print(table)


@app.command()
def create(
    variant: str = typer.Option(..., "--variant", "-v", help="Sandbox variant."),
    name: str | None = typer.Option(None, help="Name for the sandbox."),
    environment: str | None = typer.Option(
        None, help="Environment (datalayer) or kernel name (jupyter)."
    ),
    gpu: str | None = typer.Option(None, help="GPU flavor for supported variants."),
    code: str | None = typer.Option(
        None, help="Code for the batch kernel (kaggle only)."
    ),
    server_url: str | None = _SERVER_URL_OPTION,
    token: str | None = _TOKEN_OPTION,
    proxy_token: str | None = _PROXY_TOKEN_OPTION,
    run_url: str | None = _RUN_URL_OPTION,
    app_name: str | None = _APP_NAME_OPTION,
    username: str | None = _USERNAME_OPTION,
) -> None:
    """Create a sandbox and leave it running, detached from this process."""
    manager_kwargs = _manager_kwargs(
        server_url, token, proxy_token, run_url, app_name, username
    )
    create_kwargs: dict[str, Any] = {}
    if gpu:
        create_kwargs["gpu"] = gpu
    if variant.strip().lower().replace("-", "_") == "jupyter":
        if environment:
            create_kwargs["kernel_name"] = environment
    elif environment:
        create_kwargs["environment"] = environment
    if code:
        create_kwargs["code"] = code
    if name:
        create_kwargs["name"] = name
    try:
        info = get_manager(variant, **manager_kwargs).create(**create_kwargs)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    table = _sandbox_table("Created")
    _add_sandbox_row(table, info)
    console.print(table)


@app.command()
def environments(
    variant: str | None = _VARIANT_OPTION,
    token: str | None = _TOKEN_OPTION,
    run_url: str | None = _RUN_URL_OPTION,
) -> None:
    """List the environments sandboxes can be created in."""
    variants = [variant] if variant else manageable_variants()
    table = Table(title="Sandbox Environments")
    table.add_column("Variant", style="magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Title")
    table.add_column("Language")
    table.add_column("Burning rate", justify="right")
    kwargs: dict[str, Any] = {}
    if token:
        kwargs["token"] = token
    if run_url:
        kwargs["run_url"] = run_url
    total = 0
    skipped: list[tuple[str, str]] = []
    for name in variants:
        try:
            environments = Sandbox.list_environments(
                variant=name, **(kwargs if name == "datalayer" else {})
            )
        except Exception as exc:
            skipped.append((name, str(exc)))
            continue
        for env in environments:
            table.add_row(
                name,
                env.name,
                env.title or "",
                env.language or "",
                str(env.burning_rate or ""),
            )
            total += 1
    if total:
        console.print(table)
    else:
        console.print("[yellow]No environments found.[/yellow]")
    for name, reason in skipped:
        console.print(f"[dim]({name} skipped: {reason})[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
