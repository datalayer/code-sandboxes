# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Typer CLI for interactive sandbox REPL sessions."""

from __future__ import annotations

import os
from typing import Any

import typer

from . import Sandbox
from .models import Result

app = typer.Typer(help="Interactive REPL for code-sandboxes variants.")

_SUPPORTED_REPL_VARIANTS = {
    "jupyter",
    "docker",
    "eval",
    "monty",
    "colab",
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

    if variant == "colab":
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
        help="Sandbox variant (jupyter, docker, eval, monty, colab, kaggle, modal, datalayer).",
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
