<!--
  ~ Copyright (c) 2025-2026 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

# Changelog

## Unreleased

- Added `ColabSandbox(use_browser_bridge=True)` to obtain the Colab runtime
  connection details (`server_url` / `kernel_id` / `proxy_token`) from an
  authenticated Colab browser session via `jupyter-kernel-client`'s browser
  bridge, instead of requiring them to be supplied manually.
- Breaking change: sandbox variant names are `eval`, `docker`, `jupyter`, and `datalayer`.
- Removed support for the older `local-*` variant names from the public API and documentation.
- Clarified in the documentation that `Sandbox.create()` defaults to `datalayer`.
