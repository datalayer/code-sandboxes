# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: e2b sandbox (Firecracker microVM with a Jupyter kernel).

Run with:
  python examples/repl/e2b_sandbox_example.py

Auth:
- create an API key at https://e2b.dev and export E2B_API_KEY.
- export E2B_DOMAIN as well to talk to a self-hosted cluster rather than to
  e2b.dev.

The prompt behaves as a REPL should: the kernel holds one namespace, so
definitions persist between lines, and a line that is an expression answers
with its value.
"""

import argparse
import os

from code_sandboxes import Sandbox, provider_ingress_execution, run_repl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the e2b sandbox REPL example.")
    parser.add_argument(
        "--template",
        default=os.environ.get("E2B_TEMPLATE"),
        help=(
            "The E2B template to create from. `code-interpreter-v1` unless "
            "told otherwise — and anything named here has to be built on top "
            "of it, since only a template carrying a Jupyter kernel can serve "
            "the interpreter this variant drives."
        ),
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=5.0,
        help=(
            "How long the sandbox may live. E2B takes one down when its "
            "timeout runs out, whatever it is doing — including a prompt "
            "somebody is still typing at."
        ),
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Execute directly through the E2B code-interpreter adapter.",
    )
    return parser.parse_args()


def _examples() -> list[tuple[str, str]]:
    """Snippets worth pasting into this sandbox, for `:examples`."""
    return [
        (
            "State is kept between lines",
            """
            totals = [1, 2, 3]
            totals.append(4)
            sum(totals)
            """,
        ),
        (
            "The one backend that answers with rich outputs: this is an image",
            """
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1, 4, 9, 16], marker="o")
            ax.set_title("returned as a PNG, not as text")
            fig
            """,
        ),
        (
            "And an HTML repr comes back as HTML",
            """
            import pandas as pd
            pd.DataFrame({"variant": ["e2b", "daytona"], "state": [True, True]})
            """,
        ),
        (
            "Where this is running",
            """
            import platform, sys
            platform.node(), platform.platform(), sys.version.split()[0]
            """,
        ),
        (
            "The filesystem is the sandbox's own",
            """
            from pathlib import Path
            Path("/tmp/notes.txt").write_text("written inside the sandbox")
            Path("/tmp/notes.txt").read_text()
            """,
        ),
    ]


def main() -> None:
    args = _parse_args()
    if not os.environ.get("E2B_API_KEY"):
        print("E2B auth not found.")
        print("Set E2B_API_KEY. Create a key at https://e2b.dev.")
        raise SystemExit(1)

    print(f"Launching e2b sandbox REPL from template: {args.template or 'code-interpreter-v1'}")

    try:
        with (
            Sandbox.create(
                variant="e2b", timeout=60, template=args.template, examples=_examples()
            ) as provider,
            provider_ingress_execution(provider, direct=args.direct) as sandbox,
        ):
            print(f"Sandbox: {provider.sandbox_id}")
            # A REPL is read at human speed, and the default life of a sandbox
            # is shorter than a session usually is.
            provider.set_timeout(args.minutes * 60)
            run_repl(sandbox)
    except Exception as exc:
        print("e2b REPL failed:", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
