# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: e2b sandbox (Firecracker microVM with a Jupyter kernel).

Run with:
  python examples/exec/e2b_sandbox_example.py

Auth:
- create an API key at https://e2b.dev and export E2B_API_KEY.
- export E2B_DOMAIN as well to talk to a self-hosted cluster rather than to
  e2b.dev.

The variant drives E2B through its code interpreter SDK, which keeps a Jupyter
kernel per context. So two things hold here that do not hold in a variant
running a process per snippet: definitions persist between snippets, and rich
display data — an HTML repr, a figure — comes back as a result rather than
being lost.
"""

import argparse
import os

from code_sandboxes import Sandbox, provider_ingress_execution, show_and_run


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the e2b sandbox example.")
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
        "--direct",
        action="store_true",
        help="Execute directly through the E2B code-interpreter adapter.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("E2B_API_KEY"):
        print("E2B auth not found.")
        print("Set E2B_API_KEY. Create a key at https://e2b.dev.")
        raise SystemExit(1)

    print(f"Launching e2b sandbox from template: {args.template or 'code-interpreter-v1'}")

    try:
        with Sandbox.create(
            variant="e2b", timeout=60, template=args.template
        ) as provider, provider_ingress_execution(
            provider, direct=args.direct
        ) as sandbox:
            print(f"Sandbox: {provider.sandbox_id}")

            # What tells this variant apart from a per-snippet runner: the
            # kernel holds one namespace, so the second snippet sees what the
            # first defined.
            show_and_run(sandbox, "x = 40")
            show_and_run(sandbox, "import sys; print(sys.version.split()[0])")
            state = show_and_run(sandbox, "x + 2")
            if state.text != "42":
                raise RuntimeError(
                    f"State did not survive between snippets: x + 2 gave {state.text!r}."
                )
            print("state verified: the namespace is shared between snippets.")

            print("-- streaming: one number should appear every second --")
            show_and_run(
                sandbox,
                "import time\nfor i in range(1, 10):\n    print(i)\n    time.sleep(1)",
            )

            # A kernel has a channel for rich display data, which a process
            # writing to stdout has not: what the code displays arrives as a
            # result keyed by its MIME type.
            rich = show_and_run(
                sandbox,
                "from IPython.display import HTML\nHTML('<b>from e2b</b>')",
            )
            if not any(result.html for result in rich.results):
                raise RuntimeError("The HTML repr did not come back as a result.")
            print("rich output verified: text/html arrived as a result.")

            # E2B takes a sandbox down when its timeout runs out, whatever it
            # is doing — so a long job says so before it starts, and the count
            # restarts at the call.
            provider.set_timeout(300)
            print("timeout extended: the sandbox has five minutes from now.")

            # Every port inside has a public host of its own, which is what
            # makes a server started in the sandbox reachable without a tunnel.
            print("host for port 8000:", provider.get_host(8000))

            # Bytes take the filesystem of the sandbox, not a program that
            # decodes them.
            provider.files.write_bytes("/tmp/hello.bin", b"from e2b")
            print("file round trip:", provider.files.read_bytes("/tmp/hello.bin"))

            # The error path, demonstrated ON PURPOSE: the run must not die,
            # the failure must come back as a `code_error` on the result. Said
            # before it happens, or the example's last lines read as a crash.
            print("-- error handling: the next snippet raises deliberately --")
            error_result = show_and_run(sandbox, "raise RuntimeError('e2b failure example')")
            if error_result.code_error is None:
                raise RuntimeError("The deliberate failure did not surface as a code_error.")
            print("error captured as expected — e2b example completed.")
    except Exception as exc:
        print("e2b example failed:", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
