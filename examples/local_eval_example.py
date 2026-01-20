# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: local-eval sandbox (no isolation).

Run with:
  python examples/local_eval_example.py
"""

from code_sandboxes import Sandbox


def main() -> None:
    with Sandbox.create(variant="local-eval", timeout=30) as sandbox:
        result = sandbox.run_code("x = 21 * 2\nprint(x)")
        print("stdout:", result.stdout)

        sandbox.files.write("/tmp/hello.txt", "Hello from local-eval")
        content = sandbox.files.read("/tmp/hello.txt")
        print("file:", content)

        cmd = sandbox.commands.run("python", "-c", "print('ok')")
        print("cmd:", cmd.stdout.strip())


if __name__ == "__main__":
    main()
