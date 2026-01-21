# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: local-jupyter sandbox (Jupyter kernel isolation with persistent state).

Run with:
  python examples/local_jupyter_example.py

Note: This requires jupyter_server and jupyter-kernel-client.
"""

from code_sandboxes import Sandbox


def main() -> None:
    try:
        with Sandbox.create(variant="local-jupyter", timeout=30) as sandbox:
            # Test persistent state across executions
            sandbox.run_code("x = 40")
            result = sandbox.run_code("x + 2")
            print("result:", result.text)  # Should print 42

            # Test stdout
            result = sandbox.run_code("print('hello from jupyter')")
            print("stdout:", result.stdout)

            # Test file operations
            sandbox.files.write("/tmp/jupyter_test.txt", "Hello from local-jupyter")
            content = sandbox.files.read("/tmp/jupyter_test.txt")
            print("file:", content)

            # Test command execution
            cmd = sandbox.commands.run("python", "-c", "print('cmd ok')")
            print("cmd:", cmd.stdout.strip())
    except ModuleNotFoundError as exc:
        print("local-jupyter sandbox is not available:", exc)
    except Exception as exc:  # noqa: BLE001
        print("local-jupyter example failed:", exc)


if __name__ == "__main__":
    main()
