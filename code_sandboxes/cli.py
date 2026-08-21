# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Typer CLI for code sandboxes: running code, and CRUD management."""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import Sandbox
from .console import run_repl, show_and_run, show_result
from .manage import SandboxManagementError, get_manager, manageable_variants
from .models import normalize_variant

app = typer.Typer(help="Code sandboxes: run a REPL, list, create and delete sandboxes.")

console = Console()

#: The variants code can be run in from here, by their canonical names. A
#: caller may spell one with an underscore or in capitals; `normalize_variant`
#: brings it back to one of these.
_SUPPORTED_RUN_VARIANTS = frozenset(
    {
        "cloudflare",
        "coreweave",
        "datalayer",
        "daytona",
        "docker",
        "e2b",
        "eval",
        "google-colab",
        "jupyter-server",
        "kaggle",
        "modal",
        "monty",
    }
)


#: The variants a GPU can be asked of. The others have none at all, and say so
#: rather than running on a CPU as though nothing had been asked.
_GPU_VARIANTS = frozenset({"coreweave", "datalayer", "daytona", "kaggle", "modal"})


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """Code sandboxes CLI."""
    if ctx.invoked_subcommand is None:
        _run_repl(variant="jupyter-server")


def _resolve_variant(variant: str | None) -> str:
    if not variant:
        variant = typer.prompt(
            "Sandbox variant",
            default="jupyter-server",
            show_default=True,
        )
    selected = normalize_variant(variant)
    if selected not in _SUPPORTED_RUN_VARIANTS:
        raise typer.BadParameter(
            f"Unsupported variant: {variant}. Supported values: "
            + ", ".join(sorted(_SUPPORTED_RUN_VARIANTS))
        )
    return selected


def _colab_kwargs(
    server_url: str | None, kernel_id: str | None, proxy_token: str | None
) -> dict[str, Any]:
    """What a Colab runtime needs, asked for whatever was not given."""
    return {
        "server_url": server_url or typer.prompt("Colab runtime URL (RUNTIME_URL)"),
        "kernel_id": kernel_id or typer.prompt("Colab kernel id (RUNTIME_ID)"),
        "proxy_token": proxy_token
        or typer.prompt(
            "Colab runtime proxy token (RUNTIME_PROXY_TOKEN)",
            hide_input=True,
        ),
    }


def _kaggle_kwargs(
    server_url: str | None, kernel_id: str | None, token: str | None
) -> dict[str, Any]:
    """What a Kaggle runtime needs; the kernel is optional, the URL is not."""
    kwargs: dict[str, Any] = {
        "server_url": server_url or typer.prompt("Kaggle runtime proxy URL (RUNTIME_URL)"),
    }
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
    return kwargs


def _resolve_variant_kwargs(  # noqa: C901
    variant: str,
    server_url: str | None,
    kernel_id: str | None,
    proxy_token: str | None,
    token: str | None,
    run_url: str | None,
    gpu: str | None,
    spot: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    if variant == "jupyter-server":
        # Match `jupyter console` behavior by launching local Jupyter on random port.
        kwargs["port"] = 0

    if variant == "google-colab":
        kwargs.update(_colab_kwargs(server_url, kernel_id, proxy_token))

    if variant == "kaggle":
        kwargs.update(_kaggle_kwargs(server_url, kernel_id, token))

    if variant == "datalayer":
        if token:
            kwargs["token"] = token
        if run_url:
            kwargs["run_url"] = run_url

    if gpu:
        # A GPU reaches the variants that can give one, and is REFUSED by the
        # rest rather than dropped: a sandbox that looks as though it asked
        # for an H100 and did not is one whose timings mean nothing.
        if variant not in _GPU_VARIANTS:
            raise typer.BadParameter(
                f"--gpu is not something {variant} can give. The variants with "
                "a GPU are: " + ", ".join(sorted(_GPU_VARIANTS)) + "."
            )
        kwargs["gpu"] = gpu

    if spot:
        if variant != "daytona":
            raise typer.BadParameter(f"--spot is a daytona option; {variant} has none.")
        kwargs["spot"] = True

    return kwargs


#: The options both ways of running code take. Declared once: `exec` and
#: `repl` open the same sandbox and differ only in what they then do with it,
#: and two copies of nine options drift.
_RUN_VARIANT_OPTION = typer.Option(
    None,
    "--variant",
    "-v",
    # Built from the set itself, the way the management commands build theirs:
    # a hand-written list is one more place to forget a variant, and it had
    # already fallen behind twice.
    help="Sandbox variant (" + ", ".join(sorted(_SUPPORTED_RUN_VARIANTS)) + ").",
)
_RUN_TIMEOUT_OPTION = typer.Option(60.0, help="Code execution timeout (seconds).")
_RUN_ENVIRONMENT_OPTION = typer.Option(
    None, help="Sandbox environment (used by variants such as datalayer)."
)
_RUN_SERVER_URL_OPTION = typer.Option(None, help="Colab/Kaggle runtime URL.")
_RUN_KERNEL_ID_OPTION = typer.Option(None, help="Colab/Kaggle kernel ID.")
_RUN_PROXY_TOKEN_OPTION = typer.Option(None, help="Colab runtime proxy token.")
_RUN_TOKEN_OPTION = typer.Option(None, help="Datalayer/Kaggle API token override.")
_RUN_RUN_URL_OPTION = typer.Option(None, help="Datalayer run URL override.")
_EXEC_CODE_ARGUMENT = typer.Argument(
    None,
    help="The code to run. Read from stdin when neither this nor --file is given.",
)
_EXEC_FILE_OPTION = typer.Option(None, "--file", "-f", help="Read the code from a file instead.")
_EXEC_QUIET_OPTION = typer.Option(
    False,
    "--quiet",
    "-q",
    help="Print only what the code produced — no banner, no echo of the code.",
)
_RUN_SPOT_OPTION = typer.Option(
    False,
    "--spot",
    help=(
        "Run on preemptible GPU capacity (daytona): far cheaper and outside "
        "the GPU quota, and reclaimed without warning. Needs --gpu."
    ),
)
_RUN_GPU_OPTION = typer.Option(
    None,
    "--gpu",
    help=(
        "GPU flavor / accelerator for supported variants "
        "(modal/datalayer examples: T4, A10G, A100, H100; "
        "daytona examples: H100, H200, RTX-4090; "
        "kaggle examples: NvidiaTeslaT4, NvidiaTeslaP100, or aliases T4/P100)."
    ),
)


@contextlib.contextmanager
def _started_sandbox(
    variant: str | None,
    timeout: float,
    environment: str | None,
    *,
    announce: bool = True,
    server_url: str | None = None,
    kernel_id: str | None = None,
    proxy_token: str | None = None,
    token: str | None = None,
    run_url: str | None = None,
    gpu: str | None = None,
    spot: bool = False,
) -> Iterator[Sandbox]:
    """A started sandbox of the variant asked for, gone again on the way out.

    Both ways of running code want the same thing — resolve the variant,
    gather what that variant needs to connect, start it, and be certain it is
    terminated afterwards. They differ only in what happens in between.

    A failure to START is reported here, because there is nothing to report
    it to yet. A failure inside the block is the caller's: it travels, so
    `exec` can end with the status its code earned.
    """
    selected = _resolve_variant(variant)
    sandbox_kwargs = _resolve_variant_kwargs(
        selected,
        server_url=server_url,
        kernel_id=kernel_id,
        proxy_token=proxy_token,
        token=token,
        run_url=run_url,
        gpu=gpu,
        spot=spot,
    )
    if announce:
        console.print(f"Starting sandbox variant: {selected}", style="cyan")
    started = False
    try:
        with Sandbox.create(
            variant=selected,
            timeout=timeout,
            environment=environment,
            **sandbox_kwargs,
        ) as sandbox:
            started = True
            if announce:
                identifier = sandbox.sandbox_id or "<unknown>"
                console.print(f"Sandbox started (id={identifier}).", style="green")
            yield sandbox
    except Exception as exc:
        if started:
            # Not ours to report: whatever the block raised travels, so that
            # `exec` can end with the status its own code earned.
            raise
        console.print(f"Failed to start the {selected} sandbox: {exc}", style="red")
        raise typer.Exit(code=1) from None
    if announce:
        console.print("Sandbox terminated.", style="green")


@app.command()
def repl(
    variant: str | None = _RUN_VARIANT_OPTION,
    timeout: float = _RUN_TIMEOUT_OPTION,
    environment: str | None = _RUN_ENVIRONMENT_OPTION,
    server_url: str | None = _RUN_SERVER_URL_OPTION,
    kernel_id: str | None = _RUN_KERNEL_ID_OPTION,
    proxy_token: str | None = _RUN_PROXY_TOKEN_OPTION,
    token: str | None = _RUN_TOKEN_OPTION,
    run_url: str | None = _RUN_RUN_URL_OPTION,
    gpu: str | None = _RUN_GPU_OPTION,
    spot: bool = _RUN_SPOT_OPTION,
) -> None:
    """Open an interactive prompt on a sandbox of the selected variant.

    State is kept between lines. The sandbox is always terminated when this
    command exits.
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
        spot=spot,
    )


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
    spot: bool = False,
) -> None:
    with _started_sandbox(
        variant,
        timeout,
        environment,
        server_url=server_url,
        kernel_id=kernel_id,
        proxy_token=proxy_token,
        token=token,
        run_url=run_url,
        gpu=gpu,
        spot=spot,
    ) as sandbox:
        run_repl(sandbox, console=console)


def _code_to_run(code: str | None, file: Path | None) -> str:
    """The snippet to run, from the argument, a file, or what was piped in."""
    if file is not None:
        if code is not None:
            raise typer.BadParameter("Give the code or --file, not both.")
        return file.read_text(encoding="utf-8")
    if code is not None:
        return code
    if sys.stdin.isatty():
        raise typer.BadParameter(
            "No code to run: pass it as an argument, with --file, or on stdin."
        )
    return sys.stdin.read()


@app.command("exec")
def exec_code(
    code: str | None = _EXEC_CODE_ARGUMENT,
    file: Path | None = _EXEC_FILE_OPTION,
    quiet: bool = _EXEC_QUIET_OPTION,
    variant: str | None = _RUN_VARIANT_OPTION,
    timeout: float = _RUN_TIMEOUT_OPTION,
    environment: str | None = _RUN_ENVIRONMENT_OPTION,
    server_url: str | None = _RUN_SERVER_URL_OPTION,
    kernel_id: str | None = _RUN_KERNEL_ID_OPTION,
    proxy_token: str | None = _RUN_PROXY_TOKEN_OPTION,
    token: str | None = _RUN_TOKEN_OPTION,
    run_url: str | None = _RUN_RUN_URL_OPTION,
    gpu: str | None = _RUN_GPU_OPTION,
    spot: bool = _RUN_SPOT_OPTION,
) -> None:
    """Run one snippet in a fresh sandbox and print what it produced.

    The sandbox is created, the code is run, and the sandbox is terminated.
    The exit status is the code's own — 0 when it ran cleanly, 1 when it
    raised or the sandbox could not run it — so this composes in a shell:

        code-sandboxes exec -v eval -q 'print(40 + 2)' | wc -l
    """
    snippet = _code_to_run(code, file)
    with _started_sandbox(
        variant,
        timeout,
        environment,
        announce=not quiet,
        server_url=server_url,
        kernel_id=kernel_id,
        proxy_token=proxy_token,
        token=token,
        run_url=run_url,
        gpu=gpu,
        spot=spot,
    ) as sandbox:
        if quiet:
            result = sandbox.run_code(snippet)
            show_result(result, console=console, labelled=False)
        else:
            result = show_and_run(sandbox, snippet, console=console)
    if not result.success:
        raise typer.Exit(code=1)


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
_SERVER_URL_OPTION = typer.Option(None, help="Server URL (jupyter, google-colab).")
_TOKEN_OPTION = typer.Option(None, help="API token (jupyter, datalayer).")
_PROXY_TOKEN_OPTION = typer.Option(None, help="Colab runtime proxy token.")
_RUN_URL_OPTION = typer.Option(None, help="Datalayer run URL override.")
_APP_NAME_OPTION = typer.Option(None, help="Modal app name (default code-sandboxes).")
_USERNAME_OPTION = typer.Option(None, help="Kaggle username override.")
_TAG_OPTION = typer.Option(None, "--tag", help="Tag as key=value, repeatable (modal).")
_CAPABILITY_OPTION = typer.Option(None, "--capability", help="Capability, repeatable (datalayer).")


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
    kwargs = _manager_kwargs(server_url, token, proxy_token, run_url, app_name, username)
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
    kwargs = _manager_kwargs(server_url, token, proxy_token, run_url, app_name, username)
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
    kwargs = _manager_kwargs(server_url, token, proxy_token, run_url, app_name, username)
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
    code: str | None = typer.Option(None, help="New code, pushed as a new version (kaggle)."),
    server_url: str | None = _SERVER_URL_OPTION,
    token: str | None = _TOKEN_OPTION,
    proxy_token: str | None = _PROXY_TOKEN_OPTION,
    run_url: str | None = _RUN_URL_OPTION,
    app_name: str | None = _APP_NAME_OPTION,
    username: str | None = _USERNAME_OPTION,
) -> None:
    """Update a sandbox: what changes depends on the variant."""
    manager_kwargs = _manager_kwargs(server_url, token, proxy_token, run_url, app_name, username)
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
    code: str | None = typer.Option(None, help="Code for the batch kernel (kaggle only)."),
    server_url: str | None = _SERVER_URL_OPTION,
    token: str | None = _TOKEN_OPTION,
    proxy_token: str | None = _PROXY_TOKEN_OPTION,
    run_url: str | None = _RUN_URL_OPTION,
    app_name: str | None = _APP_NAME_OPTION,
    username: str | None = _USERNAME_OPTION,
) -> None:
    """Create a sandbox and leave it running, detached from this process."""
    manager_kwargs = _manager_kwargs(server_url, token, proxy_token, run_url, app_name, username)
    create_kwargs: dict[str, Any] = {}
    if gpu:
        create_kwargs["gpu"] = gpu
    if variant.strip().lower().replace("-", "_") == "jupyter_server":
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
