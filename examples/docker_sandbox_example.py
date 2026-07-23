# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: docker sandbox (container isolation).

Run with:
    python examples/docker_sandbox_example.py

Note: This requires Docker support and the `datalayer/code-sandboxes:latest` image.
Build it with: make -C .. build-docker
"""

from code_sandboxes import Sandbox


def main() -> None:
    try:
        with Sandbox.create(
            variant="docker",
            timeout=30,
            image="code-sandboxes-jupyter:latest",
        ) as sandbox:
            result = sandbox.run_code("print('hello from docker')")
            print("stdout:", result.stdout)
            error_result = sandbox.run_code("raise RuntimeError('boom')")
            if error_result.code_error:
                print(
                    "code_error:",
                    f"{error_result.code_error.name}: {error_result.code_error.value}",
                )
            cmd = sandbox.commands.run("python", "-c", "print(123)")
            print("cmd:", cmd.stdout.strip())
    except ModuleNotFoundError as exc:
        print("docker sandbox is not available:", exc)
    except Exception as exc:
        print("docker example failed:", exc)


if __name__ == "__main__":
    main()
