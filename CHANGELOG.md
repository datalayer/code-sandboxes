<!--
  ~ Copyright (c) 2025-2026 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

# Changelog

## Unreleased

- Added `:examples` to the sandbox prompt. `run_repl(sandbox, examples=[...])`
  takes title-and-code pairs and prints them on request, for a reader to copy
  into the prompt; every REPL example under `examples/repl` ships its own, and
  the ones that can take a GPU offer device discovery and a timed matmul
  instead of their general set when `--gpu` was asked for. The snippets avoid
  blocks on purpose: the prompt reads one line at a time, so a pasted `for` or
  `def` would arrive without its body.

- Fixed a `daytona` GPU sandbox failing to be created at all unless it was
  also asking for preemptible capacity. Daytona requires every GPU sandbox to
  be ephemeral — *"GPU sandboxes must be ephemeral; set autoDeleteInterval to
  0"* — and `auto_delete_interval=0` was being set only on the `spot=True`
  path, so a plain `gpu="H100"` was refused by the API. It now follows the GPU
  itself, which is what Daytona ties it to.

- Added three cloud variants: `e2b`, `coreweave` and `cloudflare`.

  `e2b` runs in a Firecracker microVM through E2B's code interpreter SDK, so it
  holds a Jupyter kernel per context — `x = 1` in one call is still there in
  the next — and answers with rich display data: a figure comes back as an
  image, an HTML repr as HTML. It needs `E2B_API_KEY` and
  `pip install code-sandboxes[e2b]`. `set_timeout()` extends the life of a
  running sandbox and `get_host(port)` gives the public host of a port inside.

  `coreweave` runs a container on CoreWeave's GPU cloud. What the SDK offers is
  `exec` — a process at a time — so a namespace is held here instead: one
  `python -u -c` session is started with the sandbox and fed JSON lines on
  stdin, the same arrangement the `modal` variant uses, and snippets share a
  namespace as they do everywhere else. A session that cannot start, or that
  goes away, drops back to a process per snippet rather than failing. It needs
  `CWSANDBOX_API_KEY` and `pip install code-sandboxes[coreweave]`.

  `cloudflare` runs a container on Cloudflare's edge. Cloudflare's own SDK is a
  Workers binding written in TypeScript, which a Python process cannot hold, so
  this variant drives the SANDBOX BRIDGE — the Worker Cloudflare publishes to
  expose the SDK over HTTP. Deploy it once with
  `npm create cloudflare -- sandbox-bridge --template=cloudflare/sandbox-sdk/bridge/worker`,
  then set `CLOUDFLARE_SANDBOX_API_URL` and `CLOUDFLARE_SANDBOX_API_KEY`. The
  bridge gives a started process nothing to write to, so each snippet runs in
  one of its own and state does not carry between calls — put what shares state
  in one snippet, or keep it in a file, which does persist. Its manager creates,
  gets and deletes; it cannot list, because the bridge has no endpoint that
  enumerates sandboxes, and says so rather than answering with an empty list.

- Hardened the three new variants against silently doing something other than
  what was asked. `cloudflare` now carries `SandboxConfig.env_vars` into every
  snippet — the bridge takes no environment when it creates a sandbox, so they
  had been accepted and dropped — refuses a `network_policy` it cannot apply
  rather than leaving a sandbox believed to be cut off connected, refuses
  `get_variable` with the reason instead of answering the misleading "no such
  variable", and serves `files.read`/`files.write` through the bridge's own
  file endpoints so they need no session at all. `coreweave` refuses the
  variable APIs when there is no session process — under `stateful=False`, or
  after one was lost — rather than reporting a successful set that vanishes
  with the process, and a snippet that runs past its timeout now has its
  session STOPPED rather than left running and changing the namespace behind a
  call that already returned.

- A GPU asked of a variant that has none is now REFUSED rather than dropped.
  `--gpu` reaches `coreweave`, `datalayer`, `daytona`, `kaggle` and `modal`,
  and `code-sandboxes exec -v e2b --gpu H100` says which variants can give one
  instead of running on a CPU as though nothing had been asked — a sandbox that
  looks as though it asked for an H100 and did not is one whose timings mean
  nothing. `--gpu` was previously accepted and silently ignored for every other
  variant.

- Corrected the module docstring of the `modal` variant, which still described
  the process-per-snippet behaviour that the session process replaced: `modal`
  keeps a namespace between snippets, and falls back to a process per snippet
  only when the session cannot be held.

- Added GPU support to the `daytona` variant. `gpu=` takes Daytona's own
  flavors, `gpu_count=` how many, and several names comma-separated are an
  ordered list of preferences Daytona falls back along — `gpu="H100,H200"`
  takes an H200 when no H100 is free. `spot=True` runs on preemptible
  capacity, which is far cheaper and outside the GPU quota; it is GPU-only and
  built from an image with `auto_delete_interval=0`, both checked before the
  request rather than left to come back as an API error.
  `DaytonaSandbox.preempted_at()` answers when a spot sandbox was reclaimed,
  and `run_code` asks on your behalf so that an eviction is not reported as a
  dropped connection. `code-sandboxes exec/repl --spot` reaches it from the
  CLI.

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
