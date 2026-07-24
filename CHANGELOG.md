<!--
  ~ Copyright (c) 2025-2026 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

# Changelog

## Unreleased

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
