<!--
  ~ Copyright (c) 2025-2026 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

# Changelog

## Unreleased

- Renamed the `google_colab` variant to `google-colab`, so every canonical
  variant name is spelled the one way (`jupyter-server` already was). Any
  spelling is still accepted everywhere a variant is named — `normalize_variant`
  now folds to the canonical dashed form rather than to underscores, which is
  what a dispatcher compares against, so the two can no longer drift apart.

- Added `code-sandboxes exec`, which runs one snippet in a fresh sandbox of any
  variant and exits with the status the code earned — `0` when it ran cleanly,
  `1` when it raised — so it composes in a shell. The code comes from an
  argument, from `--file`, or from standard input; `--quiet` prints only what
  the code produced. `exec` and `repl` take the same options.

- Moved the machinery for showing a run — `show_code`, `show_result`,
  `show_and_run`, `run_repl`, `repl_prompt` — into `code_sandboxes.console`,
  exported from the package. It existed three times over: in the CLI, in the
  REPL examples and in the exec examples, disagreeing about whether the value
  of a trailing expression is shown, whether stderr is told apart from stdout,
  and which words end a session. The examples now import it like any other
  consumer, and `examples/*/[exec|repl]_common.py` are gone.

- Added the `daytona` sandbox variant (`DaytonaSandbox`), running code in a
  [Daytona](https://www.daytona.io/docs/) cloud sandbox. It drives the
  sandbox's code interpreter rather than `process.code_run`, so state persists
  between calls and `create_context()` gives a namespace Daytona keeps apart.
  The value of a trailing expression is captured and returned as
  `ExecutionResult.text`, which the interpreter itself does not report. GPUs,
  cpu/memory and the network policy map onto Daytona's own settings; binary
  files go through its filesystem API. Authenticate with `DAYTONA_API_KEY` (or
  `DAYTONA_JWT_TOKEN` with `DAYTONA_ORGANIZATION_ID`) and install with
  `pip install code-sandboxes[daytona]`. `get_manager("daytona")` answers the
  CRUD verbs over an organization's sandboxes.

- Added the `kaggle` sandbox variant (`KaggleSandbox`) to connect to a Kaggle
  interactive notebook runtime via `jupyter-kernel-client`'s
  `KaggleKernelClient`. Authenticate with a Kaggle API token (`token` argument or
  the `KAGGLE_API_TOKEN` environment variable) — omitting `kernel_id` then creates
  a new kernel. Alternatively, connect to an existing session with a
  `server_url`/`kernel_id` or a notebook session `channels_url` (the signed JWT in
  the proxied URL provides the authentication). Install with
  `pip install code-sandboxes[kaggle]`.

- Enhanced `KaggleSandbox` with a transparent batch primitive: when no runtime
  connection details are provided, it automatically executes code through
  `KaggleKernelExecutor` (submit/poll/download) so integrations like
  `jupyter-mcp-server` can run on Kaggle without requiring interactive runtime
  wiring.

- Added Kaggle accelerator forwarding in batch mode: `Sandbox.create(variant="kaggle", gpu=...)`
  now passes the value to `KaggleKernelExecutor.execute(accelerator=...)`,
  supporting both Kaggle API values (`NvidiaTeslaT4`, ...) and friendly aliases
  (`T4`, `P100`, ...).

- Updated `ColabSandbox` to be reuse-only for existing Colab runtimes and added
  `channels_url` parsing support for extracting `server_url` / `kernel_id` /
  `proxy_token` directly from the Colab WebSocket channels URL.

- Breaking change: sandbox variant names are `eval`, `docker`, `jupyter`, and `datalayer`.

- Removed support for the older `local-*` variant names from the public API and documentation.

- Clarified in the documentation that `Sandbox.create()` defaults to `datalayer`.
