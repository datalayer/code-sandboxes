# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: docker sandbox (container isolation).

Run with:
    python examples/docker_sandbox_example.py

Note: This requires Docker support and the `datalayer/code-sandboxes:latest` image.
Build it with: make -C .. build-docker
"""

from code_sandboxes import Sandbox, show_and_run


def main() -> None:
    try:
        with Sandbox.create(
            variant="docker",
            timeout=30,
            image="code-sandboxes-jupyter:latest",
        ) as sandbox:
            show_and_run(sandbox, "print('hello from docker')")
            error_result = show_and_run(sandbox, "raise RuntimeError('boom')")
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
