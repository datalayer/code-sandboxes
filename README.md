<!--
  ~ Copyright (c) 2025-2026 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

[![Datalayer](https://assets.datalayer.tech/datalayer-25.svg)](https://datalayer.io)

[![Become a Sponsor](https://img.shields.io/static/v1?label=Become%20a%20Sponsor&message=%E2%9D%A4&logo=GitHub&style=flat&color=1ABC9C)](https://github.com/sponsors/datalayer)
[![PyPI - Version](https://img.shields.io/pypi/v/code-sandboxes)](https://pypi.org/project/code-sandboxes)

# { } 📦 Code Sandboxes

Code Sandboxes (`code_sandboxes`) is a Python package for running code in isolated sandbox variants through a unified API.

Canonical variant names:

- `datalayer`
- `docker`
- `eval`
- `google_colab`
- `jupyter`
- `kaggle`
- `daytona`
- `modal`
- `monty`

## Documentation

The full documentation is the single source of truth:

- Docs home: [https://code-sandboxes.datalayer.tech](https://code-sandboxes.datalayer.tech)
- Sandboxes and variant setup: [https://code-sandboxes.datalayer.tech/sandboxes](https://code-sandboxes.datalayer.tech/sandboxes)
- Installation: [https://code-sandboxes.datalayer.tech/installation](https://code-sandboxes.datalayer.tech/installation)
- CLI usage: [https://code-sandboxes.datalayer.tech/cli](https://code-sandboxes.datalayer.tech/cli)
- API reference: [https://code-sandboxes.datalayer.tech/api-reference](https://code-sandboxes.datalayer.tech/api-reference)
- Examples: [https://code-sandboxes.datalayer.tech/examples](https://code-sandboxes.datalayer.tech/examples)
- Comparison: [https://code-sandboxes.datalayer.tech/comparison](https://code-sandboxes.datalayer.tech/comparison)

Published site:

- [https://code-sandboxes.datalayer.tech](https://code-sandboxes.datalayer.tech)

## Install

```bash
pip install code-sandboxes
```

For backend-specific extras and credentials, see [https://code-sandboxes.datalayer.tech/installation](https://code-sandboxes.datalayer.tech/installation) and [https://code-sandboxes.datalayer.tech/sandboxes](https://code-sandboxes.datalayer.tech/sandboxes).

### Jupyter Sandbox

```python
from code_sandboxes import Sandbox

# Option 1: manage a local Jupyter server automatically
with Sandbox.create(variant="jupyter") as sandbox:
  print(sandbox.run_code("1 + 1").text)  # 2

# Option 2: connect to an existing Jupyter server
with Sandbox.create(
  variant="jupyter",
  server_url="http://localhost:8888",
  token="MY_TOKEN",
) as sandbox:
  sandbox.run_code("x = 40")
  print(sandbox.run_code("x + 2").text)  # 42
```

## Kaggle Sandbox

Kaggle supports both batch execution and interactive connections through the
`kaggle` sandbox. Install its optional dependency first:

```bash
pip install "code-sandboxes[kaggle]"
```

Required credentials for batch mode:

- `~/.kaggle/kaggle.json`, or
- `KAGGLE_API_KEY`

```bash
# Install Kaggle support
pip install code-sandboxes[kaggle]

# Optional: env-based credentials (if not using ~/.kaggle/kaggle.json)
export KAGGLE_API_KEY="<your-kaggle-api-key>"

# Launch the REPL
sandbox repl --variant kaggle
```

For batch execution, configure Kaggle credentials and create the sandbox
without a runtime URL:

```python
from code_sandboxes import Sandbox

with Sandbox.create(variant="kaggle") as sandbox:
    result = sandbox.run_code("print('hello from kaggle')")
    print(result.stdout)
```

The lower-level batch API is also available directly:

```python
from code_sandboxes import KaggleKernelExecutor

executor = KaggleKernelExecutor()
result = executor.execute(
    "print('hello from kaggle')",
    title="code-sandboxes-demo",
    accelerator="NvidiaTeslaT4",
    wait=True,
)
print(result.status, result.stdout)
print(result.to_kernel_reply())
```

For interactive execution, copy the WebSocket channels URL from an active
Kaggle notebook session and pass it to the sandbox or client:

```python
from code_sandboxes import KaggleKernelClient

with KaggleKernelClient.from_channels_url(channels_url, token=None) as kernel:
    print(kernel.execute("x = 1 + 1; print(x)"))
```

See the [complete Kaggle guide](https://code-sandboxes.datalayer.tech/sandboxes/kaggle) for authentication,
accelerators, channels URL retrieval, and execution options.

## Google Colab

Google Colab exposes an already-running kernel through an authenticating proxy.
Copy its WebSocket channels URL from the browser's Network tools, then pass it
directly to the sandbox:

```python
from code_sandboxes import Sandbox

with Sandbox.create(variant="google_colab", channels_url=channels_url) as sandbox:
    print(sandbox.run_code("x = 1 + 1; print(x)").stdout)
```

The lower-level client and parser are owned by Code Sandboxes as well:

```python
from code_sandboxes import GoogleColabKernelClient, parse_google_colab_channels_url

server_url, kernel_id, proxy_token = parse_google_colab_channels_url(channels_url)
with GoogleColabKernelClient.from_channels_url(channels_url) as kernel:
    print(kernel.execute("print('hello from colab')"))
```

See the [complete Google Colab guide](https://code-sandboxes.datalayer.tech/sandboxes/google-colab) for
proxy authentication, explicit connection values, and channels URL retrieval.

## Manage Sandboxes (CRUD)

Every variant answers the same verbs — create, list, get, update, delete —
from Python or from the CLI, rendered as [rich](https://rich.readthedocs.io/)
tables:

```bash
code-sandboxes list                 # every variant that answers, one table
code-sandboxes list -v kaggle       # one variant
code-sandboxes get <id> -v modal    # one sandbox, live status
code-sandboxes create -v modal      # create detached, leave it running
code-sandboxes update <id> -v modal --tag team=ai   # tags (modal), --name
                                    # (docker), --capability (datalayer),
                                    # --code (kaggle: a new version)
code-sandboxes delete <id> -v modal --yes
code-sandboxes environments         # what sandboxes can be created in
```

```python
from code_sandboxes import get_manager

manager = get_manager("modal")
for info in manager.list():
    print(info.id, info.status)
manager.delete("sb-...")
```

See the [management guide](https://code-sandboxes.datalayer.tech/cli/management)
for what each variant maps to and its connection settings.

## License

BSD 3-Clause License
