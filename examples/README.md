<!--
  ~ Copyright (c) 2025-2026 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

[![Datalayer](https://assets.datalayer.tech/datalayer-25.svg)](https://datalayer.io)

[![Become a Sponsor](https://img.shields.io/static/v1?label=Become%20a%20Sponsor&message=%E2%9D%A4&logo=GitHub&style=flat&color=1ABC9C)](https://github.com/sponsors/datalayer)

# { } Code Sandboxes Examples

This folder now contains two example sets:

- `exec/`: one-shot execution examples (run a predefined script and exit).
- `repl/`: interactive REPL examples (run ad-hoc code in a loop).

Supported sandbox variants:

- `jupyter`
- `docker`
- `eval`
- `monty`
- `colab`
- `modal`
- `datalayer`

Run one-shot examples from `examples/exec/`:

```bash
cd exec
python eval_sandbox_example.py
python jupyter_sandbox_example.py
python docker_sandbox_example.py
python monty_sandbox_example.py
python colab_sandbox_example.py
python modal_sandbox_example.py
python datalayer_sandbox_example.py
```

Or run one-shot examples via Make targets:

```bash
cd exec
make eval
make jupyter
make docker
make monty
make colab
make modal
make datalayer
```

Run REPL examples from `examples/repl/`:

```bash
cd repl
make eval
make jupyter
make docker
make monty
make colab
make modal
make datalayer
```

Notes by variant:

- `docker`: requires Docker support and a Docker image (for example `code-sandboxes-jupyter:latest`).
- `monty`: requires `code-sandboxes[monty]` (`pydantic-monty`).
- `colab`: requires `RUNTIME_URL`, `RUNTIME_ID`, and `RUNTIME_PROXY_TOKEN`.
- `modal`: requires `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` or `~/.modal.toml`.
- `datalayer`: requires Datalayer runtime credentials/config.
