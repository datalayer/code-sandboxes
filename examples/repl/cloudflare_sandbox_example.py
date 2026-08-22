# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: cloudflare sandbox (container on Cloudflare, through the bridge).

Run with:
  python examples/repl/cloudflare_sandbox_example.py

Auth:
- deploy the sandbox bridge Worker once, since Cloudflare's own SDK is a
  Workers binding a Python process cannot hold:

    npm create cloudflare -- sandbox-bridge \\
        --template=cloudflare/sandbox-sdk/bridge/worker

- export CLOUDFLARE_SANDBOX_API_URL (where the Worker answers) and
  CLOUDFLARE_SANDBOX_API_KEY (the secret it generated).

This prompt is NOT a REPL in the usual sense, and says so before it opens:
each line runs in a process of its own, so `x = 40` on one line leaves nothing
behind for `x + 2` on the next. Write what shares state on a single line, or
keep it in a file — the filesystem of the sandbox does persist.
"""

import os

from code_sandboxes import Sandbox, run_repl

BRIDGE_DEPLOY_COMMAND = (
    "npm create cloudflare -- sandbox-bridge --template=cloudflare/sandbox-sdk/bridge/worker"
)


def _examples() -> list[tuple[str, str]]:
    """Snippets worth pasting into this sandbox, for `:examples`."""
    return [
        (
            "This backend is STATELESS — the second line cannot see the first",
            """
            x = 21
            """,
        ),
        (
            "…so it fails. Send what shares state as ONE snippet instead",
            """
            x = 21
            x * 2
            """,
        ),
        (
            "Or keep it in a file: the filesystem DOES persist between snippets",
            """
            from pathlib import Path
            Path("/workspace/total.txt").write_text("42")
            """,
        ),
        (
            "…and the next snippet reads it back",
            """
            from pathlib import Path
            int(Path("/workspace/total.txt").read_text())
            """,
        ),
        (
            "Where this is running",
            """
            import platform, sys
            platform.node(), platform.platform(), sys.version.split()[0]
            """,
        ),
    ]


def main() -> None:
    if not os.environ.get("CLOUDFLARE_SANDBOX_API_URL") or not os.environ.get(
        "CLOUDFLARE_SANDBOX_API_KEY"
    ):
        print("Cloudflare sandbox bridge settings not found.")
        print("Set CLOUDFLARE_SANDBOX_API_URL and CLOUDFLARE_SANDBOX_API_KEY.")
        print("Deploy a bridge Worker with:")
        print(f"  {BRIDGE_DEPLOY_COMMAND}")
        print("See https://developers.cloudflare.com/sandbox/bridge/.")
        raise SystemExit(1)

    print("Launching cloudflare sandbox REPL through the bridge at")
    print(f"  {os.environ['CLOUDFLARE_SANDBOX_API_URL']}")

    try:
        with Sandbox.create(variant="cloudflare", timeout=60, examples=_examples()) as sandbox:
            print(f"Sandbox: {sandbox.sandbox_id}")
            # Said before the prompt opens rather than discovered at the first
            # NameError: nothing defined on one line reaches the next.
            print("Each line runs in its own process — definitions do NOT persist.")
            print("Use one line for what shares state, or a file: `open('/workspace/x', 'w')`.")
            run_repl(sandbox)
    except Exception as exc:
        print("cloudflare REPL failed:", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
