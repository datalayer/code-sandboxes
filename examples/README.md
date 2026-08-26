<!--
  ~ Copyright (c) 2025-2026 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

[![Datalayer](https://assets.datalayer.tech/datalayer-25.svg)](https://datalayer.io)

[![Become a Sponsor](https://img.shields.io/static/v1?label=Become%20a%20Sponsor&message=%E2%9D%A4&logo=GitHub&style=flat&color=1ABC9C)](https://github.com/sponsors/datalayer)

# { } 📦 Code Sandboxes Examples

This folder contains two example sets:

- `exec/`: one-shot execution examples (run a predefined script and exit).
- `repl/`: interactive REPL examples (run ad-hoc code in a loop).

Neither set carries machinery of its own. Both import it from the package —
`show_and_run` prints a snippet, runs it and prints what came back;
`run_repl` holds a prompt open — which is the same code behind
`code-sandboxes exec` and `code-sandboxes repl`. That is the point of an
example here: it shows how to use a sandbox, not how to print things.

Supported sandbox variants:

- `cloudflare`
- `coreweave`
- `datalayer`
- `daytona`
- `docker`
- `e2b`
- `eval`
- `google-colab`
- `jupyter-server`
- `kaggle`
- `modal`
- `monty`

Run one-shot examples from `examples/exec/`:

```bash
cd exec
python cloudflare_sandbox_example.py
python coreweave_sandbox_example.py
python datalayer_sandbox_example.py
python daytona_sandbox_example.py
python docker_sandbox_example.py
python e2b_sandbox_example.py
python eval_sandbox_example.py
python google_colab_sandbox_example.py
python jupyter_server_sandbox_example.py
python kaggle_sandbox_example.py
python modal_sandbox_example.py
python monty_sandbox_example.py
```

Or run one-shot examples via Make targets:

```bash
cd exec
make cloudflare
make coreweave
make datalayer
make daytona
make docker
make e2b
make eval
make google-colab
make jupyter-server
make kaggle
make modal
make monty
```

The Daytona, E2B, and Modal exec examples include a timed loop that prints the
numbers 1 through 9 one per second. Each number is rendered as it arrives; the
example does not wait for the loop to finish before displaying its output.

Daytona, E2B, and Modal use Jupyter over provider ingress by default. Use
`--direct` to execute through the provider SDK adapter instead:

```bash
# Python entry points accept the flag directly.
python daytona_sandbox_example.py --direct
python e2b_sandbox_example.py --direct
python modal_sandbox_example.py --direct

# GNU Make consumes command-line options itself, so pass example flags through
# the ARGS variable rather than writing `make daytona --direct`.
make daytona ARGS=--direct
make e2b ARGS=--direct
make modal ARGS=--direct
```

Run REPL examples from `examples/repl/`:

```bash
cd repl
make cloudflare
make coreweave
make datalayer
make daytona
make docker
make e2b
make eval
make google-colab
make jupyter-server
make kaggle
make modal
make monty
```

The REPL entry points use the same mode selection:

```bash
python daytona_sandbox_example.py --direct
make daytona ARGS=--direct
```

Notes by variant:

- `cloudflare`: requires `code-sandboxes[cloudflare]` and a deployed sandbox
  bridge Worker — `npm create cloudflare -- sandbox-bridge --template=cloudflare/sandbox-sdk/bridge/worker` — named by
  `CLOUDFLARE_SANDBOX_API_URL` with the `CLOUDFLARE_SANDBOX_API_KEY` it
  generated. Each snippet runs in a process of its own, so nothing crosses
  between them.
- `coreweave`: requires `code-sandboxes[coreweave]` and `CWSANDBOX_API_KEY`
  (`CWSANDBOX_BASE_URL` for another control plane).
- `datalayer`: requires Datalayer runtime credentials/config.
- `daytona`: requires `code-sandboxes[daytona]` and `DAYTONA_API_KEY` (or
  `DAYTONA_JWT_TOKEN` with `DAYTONA_ORGANIZATION_ID`).
- `docker`: requires Docker support and a Docker image (for example `code-sandboxes-jupyter:latest`).
- `e2b`: requires `code-sandboxes[e2b]` and `E2B_API_KEY` (`E2B_DOMAIN` for a
  self-hosted cluster).
- `google-colab`: requires `RUNTIME_URL`, `RUNTIME_ID`, and `RUNTIME_PROXY_TOKEN`.
- `kaggle`: requires `RUNTIME_CHANNELS_URL`, or `RUNTIME_URL` and `RUNTIME_ID`.
- `modal`: requires `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` or `~/.modal.toml`.
- `monty`: requires `code-sandboxes[monty]` (`pydantic-monty`).
