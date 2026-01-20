# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: local-docker sandbox (container isolation).

Run with:
  python examples/local_docker_example.py

Note: This requires Docker support and the `datalayer/code-sandboxes:latest` image.
Build it with: make -C .. build-docker
"""

from code_sandboxes import Sandbox


def main() -> None:
    try:
        with Sandbox.create(
            variant="local-docker",
            timeout=30,
            image="datalayer/code-sandboxes:latest",
        ) as sandbox:
            result = sandbox.run_code("print('hello from docker')")
            print("stdout:", result.stdout)
            result = sandbox.run_code("fail")
            print("stderr:", result.error)
            cmd = sandbox.commands.run("python", "-c", "print(123)")
            print("cmd:", cmd.stdout.strip())
    except ModuleNotFoundError as exc:
        print("local-docker sandbox is not available:", exc)
    except Exception as exc:  # noqa: BLE001
        print("local-docker example failed:", exc)


if __name__ == "__main__":
    main()
