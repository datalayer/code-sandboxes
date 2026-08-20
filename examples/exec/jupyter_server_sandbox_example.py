# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: jupyter sandbox (Jupyter kernel isolation with persistent state).

Run with:
    python examples/jupyter_server_sandbox_example.py

Note: This requires jupyter_server and jupyter-kernel-client.
"""

from exec_common import show_and_run

from code_sandboxes import Sandbox


def main() -> None:
    try:
        with Sandbox.create(variant="jupyter-server", timeout=30) as sandbox:
            # Test persistent state across executions
            show_and_run(sandbox, "x = 40")
            result = show_and_run(sandbox, "x + 2")

            # Test stdout
            result = show_and_run(sandbox, "print('hello from jupyter')")

            # Test file operations
            sandbox.files.write("/tmp/jupyter_test.txt", "Hello from jupyter")
            content = sandbox.files.read("/tmp/jupyter_test.txt")
            print("file:", content)

            # Test command execution
            cmd = sandbox.commands.run("python", "-c", "print('cmd ok')")
            print("cmd:", cmd.stdout.strip())
    except ModuleNotFoundError as exc:
        print("jupyter sandbox is not available:", exc)
    except Exception as exc:
        print("jupyter example failed:", exc)


if __name__ == "__main__":
    main()
