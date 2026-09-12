<!--
  ~ Copyright (c) 2025-2026 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

# Changelog

## Unreleased

## 1.8.0

- **The Datalayer builder** (`environments.adapters.datalayer`, PLAN_ENV.md
  E1-07). The Dockerfile is generated from the lock — `uv pip sync
  --require-hashes`, apt at the versions the lock recorded, `env` before
  anything installs, `postInstall` as uid 1000 with no network — and the push
  is by digest with an SBOM and provenance attestation, under the operability
  tag `v<n>-<build_uid>` so a retried build cannot collide with the attempt
  before it. `inspect`, `resolve`, `exists` and `delete` go through the ECR
  API. Needs the `environments-builder` extra.
- **The scan and the signature** (`environments.policy`,
  `environments.attest`, E1-08, E1-09). A critical finding with a fixed
  version blocks and one nothing fixes is recorded; the decision record keeps
  the threshold it was decided under, so what stopped a build reads a month
  later. cosign signs only once the scan passed, and a replay finds the
  signature rather than pushing a second.
- **A sandbox launches from an artifact** (E2-02): `Sandbox.create(artifact=…)`
  hands each variant its own argument — an E2B template build, a Daytona
  snapshot, a Modal image id through `Image.from_id`.
- **Fixed, and a live defect:** Daytona's adapter took the image branch
  whenever resources were requested, so a snapshot asked for with `cpu=` came
  up from a plain Debian image running none of the snapshot's content, with
  nothing saying so. The combination is refused (correction 13).

## 1.7.0

- **A version is resolved into one lock** (`code_sandboxes.environments.resolve`,
  PLAN_ENV.md E1-04, D-9). Datalayer's protected constraints are merged over the
  user's requirements — a requirement that agrees with a pin is dropped for it,
  one that contradicts it is `DL_ENV_PROTECTED_PACKAGE` with the supported range
  — the base is resolved to a digest per requested variant, and uv's refusals
  are read into the error taxonomy: a conflict with the pair that cannot hold, a
  package no index has, a protected pin, and `DL_ENV_PROVIDER_ERROR`, which is
  retryable, for a failure that is not about the version.

  The lock is uv's hashed output with the apt pins and the protected pins above
  it as comments: one document that says everything a build installs, and still
  a requirements file. `BuildkitResolveRunner` is D-9's solve, `FROM` the
  resolved base digest; `LocalResolveRunner` runs `uv` where it is called, for
  `plane local`, and refuses to pin apt rather than lock another
  distribution's versions.

- **Every variant has a real interrupt now, and can be reached to give it.**
  `Sandbox.interrupt` is two gates — `_executing_event` must be set, then
  `_do_interrupt` must answer — and seven variants failed one of them, each in
  a way that looked like success:

  - `docker`, `google_colab`, `kaggle` and `monty` had no `_do_interrupt`, so
    the base default ran: set a flag, return `True`, stop nothing. `True` means
    "the interrupt was delivered", and nothing had been. `google_colab` and
    `kaggle` did read the flag, but only *after* the run, to label a finished
    result as `interrupted` — labelling a run is not stopping one.
  - `cloudflare`, `coreweave`, `daytona`, `e2b` and `modal` answered honestly
    (`False`: this provider takes no interrupt) but never marked their execution
    window, so `interrupt()` returned at the first gate and their answer was
    never reached — and `is_executing` was always False, which every status
    above them believed.

  Docker, Colab and Kaggle now interrupt for real, delegating to the
  `JupyterKernelClient` that already knows how to authenticate to their server
  rather than rebuilding the request. Monty answers `False` with its reason:
  `feed_run` executes a snippet in one blocking call with no limits and no
  callable hook, so there is no point at which a flag could be read.

  The execution window is `@marks_execution` on `run_code` rather than a line
  in each body, because it has to close on *every* path out — including the
  early `return ExecutionResult(...)` each adapter uses for an infrastructure
  failure — and a `finally` in the decorator cannot be forgotten in one branch
  of one variant. Both invariants are asserted across the package with no
  pinned exceptions left.

- A Datalayer sandbox says when it is running code, so an interrupt can reach
  it. `Sandbox.interrupt` refuses before it delegates — `if not
  self._executing_event.is_set(): return False` — and `run_code` never set that
  event, so `is_executing` was always False, every interrupt was refused at the
  door, and the refusal was reported as "no code was running" to a caller
  watching a cell run.

  This is the other half of the interrupt fix released in 1.4.4, and the
  measurement said so: with `_do_interrupt` implemented and deployed, cancelling
  a two-minute cell ten seconds in still left the kernel busy for **60.5
  seconds**. Implementing the interrupt changed nothing while nothing could call
  it. Both are needed, and the package's tests now hold them together.

  Five variants — cloudflare, coreweave, daytona, e2b, modal — implement
  `_do_interrupt` and never mark execution either, so their interrupts are
  unreachable in the same way. Pinned as known rather than fixed blind, since
  none can be measured from here.

- `DatalayerSandbox` can be interrupted. It neither implemented `_do_interrupt`
  nor read the flag the base class sets, so the default ran instead: it sets a
  flag, returns `True`, and stops nothing. Everything above believed it —
  `tasks/cancel` in the MCP gateway marks a task `cancelled` and calls the
  interrupt `execute_cell` registered, which is this. Measured on prod1 on
  2026-09-07: a two-minute cell cancelled ten seconds in answered `cancelled`
  from both `tasks/cancel` and `tasks/get`, and the next `execute_code` on that
  session took **60.8 seconds** and came back empty, where a free kernel answers
  in about a second. The cell ran to completion on a runtime that went on being
  billed. It now delegates to the runtime's own `sandbox_client` — the
  jupyter-server client `run_code` already executes through, which interrupts
  the kernel over the REST API — and answers whether the interrupt was
  delivered, never raising: a cancel that cannot reach the kernel is a cancel
  that failed, and the caller decides what that means.

  `docker` and `monty` have the same hole and are pinned as known in
  `tests/test_a_datalayer_sandbox_can_be_interrupted.py`, so the set cannot grow
  quietly. (`google_colab` and `kaggle` are entitled to the default: they poll
  `_interrupt_requested` and stop cooperatively.)

- `provider_catalog` takes a `names` argument, so a caller can describe the
  providers it serves and pay for only those. Added under 1.3.1 without a
  version bump, which is what broke the operator: its image installs this
  package unpinned, PyPI's newest was 1.3.0, and the two-argument call landed
  on a one-argument function — `TypeError: provider_catalog() takes from 0 to 1 positional arguments but 2 were given`, and `datalayer envs ls` answered 500.
  A new public parameter is a feature; released as 1.4.0 so a dependant can ask
  for it.

- `CodeExecutionOutcome` now carries `outputs`: the rich results as Jupyter
  outputs, with their mime bundles intact. It called itself a faithful superset
  of the raw `ExecutionResult` and was not — every representation but
  `text/plain` was dropped on the way through, so a matplotlib figure reached
  its callers as the string `<Figure size 640x480 with 1 Axes>` and anything
  wanting to draw it had nothing to draw. `results` is unchanged, for callers
  that only print.

- Exported the lifecycle vocabulary from the package root, so a consumer writes
  `from code_sandboxes import SandboxLifecycle` rather than reaching into
  `code_sandboxes.lifecycle` — the import path is the part that cannot be
  changed afterwards, which is why it moved before anything depended on it.

  The vocabulary gained `update` (the Runtimes API's `PUT`) and split in two.
  `SandboxLifecycle` is one sandbox — `start`, `stop`, `pause`, `resume`,
  `snapshot`, `run_code`; `SandboxManagerLifecycle` is whoever hands them out —
  `create`, `list`, `get`, `update`. They were one protocol that quietly
  disagreed with `LIFECYCLE_OPERATIONS`, because `Sandbox.create` is a
  classmethod and a client's `create` is not; `INSTANCE_OPERATIONS` and
  `MANAGER_OPERATIONS` now say which verb belongs to which shape, and a test
  holds them to covering every verb exactly once.

  It also gained the URL builders — `runtimes_url`, `runtime_url`,
  `runtime_pause_url`, `runtime_resume_url`, `sandbox_snapshots_url`,
  `sandbox_snapshot_url`, `runtime_checkpoints_url` — so every Python caller of
  the Runtimes API builds a path from one place instead of its own f-string.
  `snapshot` is recorded against `POST /sandbox-snapshots`, which is the route
  that exists; it had been documented as a sub-path of the runtime, which was
  not.

- Numbered the prompt's examples, and made one runnable by its number:
  `:examples` lists them `1.`, `2.`, … and `:examples:2` prints the second and
  then executes it, for a reader who wants the answer rather than the paste.
  They are now declared where the sandbox is made —
  `Sandbox.create(..., examples=[...])`, carried on `SandboxConfig` — so
  `run_repl(sandbox)` finds them without being told twice; passing them to
  `run_repl` still overrides for one prompt. Snippets are printed with Rich's
  markup off, since `[...]` was being read as a style tag and a snippet
  holding `list[str]` printed as `list = []`, wrong exactly where someone was
  about to copy it.

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
