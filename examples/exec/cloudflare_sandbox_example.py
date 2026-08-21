# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: cloudflare sandbox (container on Cloudflare, through the bridge).

Run with:
  python examples/exec/cloudflare_sandbox_example.py

Auth:
- Cloudflare's own SDK is a Workers binding written in TypeScript, and a Python
  process cannot hold one. What a Python process can talk to is the SANDBOX
  BRIDGE — a reference-implementation Worker Cloudflare publishes, which
  exposes the SDK over HTTP. Deploy it once:

    npm create cloudflare -- sandbox-bridge \\
        --template=cloudflare/sandbox-sdk/bridge/worker

- export CLOUDFLARE_SANDBOX_API_URL (where the Worker answers, e.g.
  https://cloudflare-sandbox-bridge.<subdomain>.workers.dev) and
  CLOUDFLARE_SANDBOX_API_KEY (the secret it generated).

The bridge's exec gives no stdin, so each snippet runs in a process of its own
and `x = 1` is gone by the next call. That is not worked around here, it is
SHOWN — along with the two ways round it: put the statements that share state
in one snippet, or keep the state in a file, since the filesystem persists.
"""

import os

from code_sandboxes import Sandbox, show_and_run

BRIDGE_DEPLOY_COMMAND = (
    "npm create cloudflare -- sandbox-bridge --template=cloudflare/sandbox-sdk/bridge/worker"
)


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

    print("Launching cloudflare sandbox through the bridge at")
    print(f"  {os.environ['CLOUDFLARE_SANDBOX_API_URL']}")

    try:
        with Sandbox.create(variant="cloudflare", timeout=60) as sandbox:
            print(f"Sandbox: {sandbox.sandbox_id}")

            show_and_run(sandbox, "import sys; print(sys.version.split()[0])")

            # STATELESS, and demonstrated rather than described: the second
            # snippet is a new process, and does not know what the first one
            # defined.
            show_and_run(sandbox, "x = 40")
            gone = show_and_run(sandbox, "x + 2")
            if gone.code_error is None or gone.code_error.name != "NameError":
                raise RuntimeError(
                    "A snippet saw the namespace of the one before it, which "
                    "the bridge does not allow — this example is out of date."
                )
            print("as expected: nothing crossed between snippets.")

            # The first way round it: whatever shares state shares a snippet.
            together = show_and_run(sandbox, "x = 40\nx + 2")
            if together.text != "42":
                raise RuntimeError(f"One snippet did not answer with its value: {together.text!r}.")
            print("one snippet: the statements that share state ran together.")

            # The second: the filesystem of the sandbox does persist, so state
            # that has to outlive a call goes there rather than in a variable.
            sandbox.files.write("/workspace/state.json", '{"x": 40}')
            kept = show_and_run(
                sandbox,
                "import json; json.load(open('/workspace/state.json'))['x'] + 2",
            )
            if kept.text != "42":
                raise RuntimeError(f"The state file did not survive: {kept.text!r}.")
            print("a file: the state crossed the call on the filesystem.")

            # The error path, demonstrated ON PURPOSE: the run must not die,
            # the failure must come back as a `code_error` on the result. Said
            # before it happens, or the example's last lines read as a crash.
            print("-- error handling: the next snippet raises deliberately --")
            error_result = show_and_run(sandbox, "raise RuntimeError('cloudflare failure example')")
            if error_result.code_error is None:
                raise RuntimeError("The deliberate failure did not surface as a code_error.")
            print("error captured as expected — cloudflare example completed.")
    except Exception as exc:
        print("cloudflare example failed:", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
