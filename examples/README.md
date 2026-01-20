<!--
  ~ Copyright (c) 2025-2026 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

# Code Sandboxes Examples

Run any example from the code-sandboxes package root:

```bash
python examples/local_eval_example.py
python examples/local_docker_example.py
python examples/datalayer_runtime_example.py
```

Notes:
- `local-docker` requires Docker support and a `LocalDockerSandbox` implementation.
- Build the image with: `docker build -t code-sandboxes-jupyter:latest -f docker/Dockerfile .`
- `datalayer-runtime` requires Datalayer runtime credentials/config.
