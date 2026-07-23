<!--
  ~ Copyright (c) 2025-2026 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

[![Datalayer](https://assets.datalayer.tech/datalayer-25.svg)](https://datalayer.io)

[![Become a Sponsor](https://img.shields.io/static/v1?label=Become%20a%20Sponsor&message=%E2%9D%A4&logo=GitHub&style=flat&color=1ABC9C)](https://github.com/sponsors/datalayer)

# { } Code Sandboxes

[![PyPI - Version](https://img.shields.io/pypi/v/code-sandboxes)](https://pypi.org/project/code-sandboxes)

Code Sandboxes `code_sandboxes` is a Python package for safe, isolated environments where an AI system can write, run, and test code without affecting the real world or the user's device.

This package provides a unified API for code execution with features like:

- **Code Execution**: Execute Python code with streaming output and rich results
- **Filesystem Operations**: Read, write, list, upload, and download files
- **Command Execution**: Run shell commands with streaming support
- **Context Management**: Maintain state across multiple executions
- **Snapshots**: Save and restore sandbox state (Datalayer runtime)
- **GPU Support**: Access GPU compute for ML workloads (Datalayer runtime)

## Sandbox Variants

Seven variants are available. Canonical variant names are `jupyter`, `docker`,
`eval`, `monty`, `colab`, `modal`, and `datalayer`. The older `local-*` names
are no longer supported.

| Variant     | Isolation                     | Use Case                         |
| ----------- | ----------------------------- | -------------------------------- |
| `jupyter`   | Process (Jupyter kernel)      | Persistent state, local/remote   |
| `docker`    | Container (Jupyter Server)    | Local isolated execution         |
| `eval`      | None (Python exec)            | Development, testing             |
| `monty`     | In-process secure interpreter | Fast, safe LLM snippets          |
| `colab`     | Google Colab runtime          | Free hosted GPU/CPU kernels      |
| `modal`     | Modal cloud container         | On-demand isolated cloud compute |
| `datalayer` | Cloud VM                      | Production, GPU workloads        |

See [Backend Setup Guides](#backend-setup-guides) below for per-variant
installation, credentials, and usage.

## Module Layout

Sandbox implementations are exposed as top-level modules:

- `code_sandboxes.jupyter_sandbox`
- `code_sandboxes.docker_sandbox`
- `code_sandboxes.eval_sandbox`
- `code_sandboxes.monty_sandbox`
- `code_sandboxes.colab_sandbox`
- `code_sandboxes.modal_sandbox`
- `code_sandboxes.datalayer_sandbox`

Example direct imports:

```python
from code_sandboxes.jupyter_sandbox import JupyterSandbox
from code_sandboxes.docker_sandbox import DockerSandbox
from code_sandboxes.eval_sandbox import EvalSandbox
from code_sandboxes.monty_sandbox import MontySandbox
from code_sandboxes.colab_sandbox import ColabSandbox
from code_sandboxes.modal_sandbox import ModalSandbox
from code_sandboxes.datalayer_sandbox import DatalayerSandbox
```

## Installation

```bash
# Basic installation
pip install code-sandboxes

# With Datalayer runtime support
pip install code-sandboxes[datalayer]

# With Docker support
pip install code-sandboxes[docker]

# With Google Colab support
pip install code-sandboxes[colab]

# With Monty (secure in-process interpreter) support
pip install code-sandboxes[monty]

# With Modal cloud sandbox support
pip install code-sandboxes[modal]

# All features
pip install code-sandboxes[all]
```

### Docker Variant Setup

The `docker` variant runs a Jupyter Server inside a Docker container and uses
`jupyter-kernel-client` to execute code.

Build the Docker image used by `DockerSandbox`:

```bash
docker build -t code-sandboxes-jupyter:latest -f docker/Dockerfile .
```

## Quick Start

### Simple Code Execution

```python
from code_sandboxes import Sandbox

# Create a sandbox with timeout
with Sandbox.create(variant="eval", timeout=60) as sandbox:
    # Execute code
    result = sandbox.run_code("x = 1 + 1")
    result = sandbox.run_code("print(x)")  # prints 2

    # Multi-statement blocks return the last expression
    result = sandbox.run_code("""
x = 10
x * 2
""")
    print(result.text)  # "20"

    # Access results
    print(result.stdout)  # "2"
```

### Cloud Execution with GPU

```python
from code_sandboxes import Sandbox

# Create a cloud sandbox with GPU
with Sandbox.create(
    variant="datalayer",
    gpu="T4",
    environment="python-gpu-env",
    timeout=300,
) as sandbox:
    sandbox.run_code("import torch")
    result = sandbox.run_code("print(torch.cuda.is_available())")
```

### Filesystem Operations

```python
with Sandbox.create() as sandbox:
    # Write files
    sandbox.files.write("/data/test.txt", "Hello World")

    # Read files
    content = sandbox.files.read("/data/test.txt")

    # List directory
    for f in sandbox.files.list("/data"):
        print(f.name, f.size)

    # Upload/download
    sandbox.files.upload("local_file.txt", "/remote/file.txt")
    sandbox.files.download("/remote/file.txt", "downloaded.txt")
```

### Command Execution

```python
with Sandbox.create() as sandbox:
    # Run a command and wait for completion
    result = sandbox.commands.run("ls -la")
    print(result.stdout)

    # Execute with streaming output
    process = sandbox.commands.exec("python", "-c", "print('hello')")
    for line in process.stdout:
        print(line, end="")

    # Install system packages
    sandbox.commands.install_system_packages(["curl", "wget"])
```

### Snapshots (Datalayer Runtime)

```python
with Sandbox.create(variant="datalayer") as sandbox:
    # Set up environment
    sandbox.install_packages(["pandas", "numpy"])
    sandbox.run_code("import pandas as pd; df = pd.DataFrame({'a': [1,2,3]})")

    # Create snapshot
    snapshot = sandbox.create_snapshot("my-setup")
    print(f"Snapshot created: {snapshot.id}")

# Later: restore from snapshot
with Sandbox.create(variant="datalayer", snapshot_name="my-setup") as sandbox:
    # State is restored
    result = sandbox.run_code("print(df)")
```

### Streaming Output

```python
from code_sandboxes import Sandbox, OutputMessage

def handle_stdout(msg: OutputMessage):
    print(f"[stdout] {msg.line}")

def handle_stderr(msg: OutputMessage):
    print(f"[stderr] {msg.line}")

with Sandbox.create() as sandbox:
    result = sandbox.run_code(
        "for i in range(5): print(f'Step {i}')",
        on_stdout=handle_stdout,
        on_stderr=handle_stderr,
    )
```

## CLI REPL

`sandbox` includes a Typer-based CLI that launches an interactive REPL
against a selected sandbox variant and always terminates created resources on exit.
The `code-sandboxes` command remains available as an alias.

```bash
sandbox repl --variant jupyter
sandbox repl --variant monty
sandbox repl --variant modal
sandbox repl --variant colab
```

If `--variant` is omitted, the CLI prompts for one.

Variant notes:

- `jupyter`: starts a managed local Jupyter server on a random port.
- `monty`: starts a Monty REPL-backed sandbox.
- `modal`: starts a Modal sandbox container.
- `colab`: prompts for runtime URL, kernel ID, and proxy token.

Exit with `:exit`, `:quit`, or Ctrl+D.

## Backend Setup Guides

Each backend has its own installation, credential, and parameter requirements.
Select a backend by passing `variant=...` to `Sandbox.create()`. The sections
below explain, for every variant, exactly **how to obtain the credentials and the
parameters** you need to pass.

### 1. Jupyter Server

Runs code out-of-process against a local or remote Jupyter Server via the Jupyter
kernel protocol (`jupyter-kernel-client`), providing process isolation and
persistent kernel state.

**Install** (included by default):

```bash
pip install code-sandboxes
```

**Parameters:**

| Parameter           | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| `server_url`        | Jupyter Server URL (default: an auto-started local server) |
| `token`             | Jupyter Server authentication token                        |
| `host` / `port`     | Bind address when the sandbox starts its own server        |
| `python_executable` | Interpreter used to launch the managed server              |

**How to obtain the `token`:**

- If you start the server yourself, you choose the token:
  ```bash
  jupyter server --port 8888 --IdentityProvider.token MY_TOKEN
  ```
- For an already-running server, print the URL + token with:
  ```bash
  jupyter server list
  # http://localhost:8888/?token=abcd1234...  :: /home/you/notebooks
  ```
  The `token=...` query value is your token. You can also pass the full URL as
  `server_url` (the `?token=...` is parsed automatically).
- If you omit `server_url` entirely, `JupyterSandbox` **starts and manages its own
  local Jupyter Server** and generates the token for you — no configuration needed.

**Usage:**

```python
from code_sandboxes import Sandbox

# Connect to an existing server:
with Sandbox.create(
        variant="jupyter",
        server_url="http://localhost:8888",
        token="MY_TOKEN",
) as sandbox:
        sandbox.run_code("x = 40")
        print(sandbox.run_code("x + 2").text)  # 42

# Or let the sandbox manage a local server automatically:
with Sandbox.create(variant="jupyter") as sandbox:
        print(sandbox.run_code("1 + 1").text)  # 2
```

### 2. Docker

Runs a Jupyter Server inside a Docker container for local, isolated execution and
connects to it with `jupyter-kernel-client`.

**Install:**

```bash
pip install code-sandboxes[docker]
```

**Prerequisites — verify Docker is installed and running:**

```bash
docker version   # must succeed (daemon reachable)
```

**Build the image** used by `DockerSandbox` (default tag
`code-sandboxes-jupyter:latest`):

```bash
docker build -t code-sandboxes-jupyter:latest -f docker/Dockerfile .
```

**Parameters** (no external credentials — the kernel `token` is generated
automatically):

| Parameter                 | Description                                                      |
| ------------------------- | ---------------------------------------------------------------- |
| `image`                   | Container image to run (default `code-sandboxes-jupyter:latest`) |
| `container_name`          | Optional fixed container name                                    |
| `host` / `container_port` | Where the in-container server is exposed                         |
| `auto_remove`             | Remove the container on stop (default `True`)                    |
| `workdir`                 | Host working directory to mount                                  |

**Usage:**

```python
from code_sandboxes import Sandbox

with Sandbox.create(
        variant="docker",
        image="code-sandboxes-jupyter:latest",
) as sandbox:
        result = sandbox.run_code("import sys; print(sys.version)")
        print(result.stdout)
```

### 3. Eval

Executes code in the host process with Python's `exec()`. No isolation — intended
for development and testing only.

**Install** (included by default):

```bash
pip install code-sandboxes
```

**Credentials / parameters:** none. There is nothing to configure.

**Usage:**

```python
from code_sandboxes import Sandbox

with Sandbox.create(variant="eval") as sandbox:
    sandbox.run_code("x = 1 + 1")
    print(sandbox.run_code("print(x)").stdout)  # 2
```

> ⚠️ `eval` shares memory with your process and provides no sandboxing. Never run
> untrusted code with it — use `monty`, `docker`, `modal`, or `datalayer` instead.

### 4. Monty

Runs code in [Monty](https://github.com/pydantic/monty), a minimal, secure Python
interpreter written in Rust (`pydantic-monty`). Monty executes a restricted
subset of Python in-process with microsecond startup and no access to the host
filesystem, environment, or network unless explicitly granted. Ideal for short,
LLM-generated snippets. Session state persists across `run_code` calls.

**Install:**

```bash
pip install code-sandboxes[monty]
```

**Credentials / parameters:** none required (fully local, in-process). Optional
constructor parameters on `MontySandbox`:

| Parameter            | How to obtain / when to use                                 |
| -------------------- | ----------------------------------------------------------- |
| `type_check`         | Set `True` to type-check code before running it             |
| `type_check_stubs`   | Provide type stub definitions when `type_check` is enabled  |
| `external_functions` | Dict of `{name: callable}` host functions the code may call |
| `limits`             | Monty `ResourceLimits` mapping (memory, stack depth, time)  |

**Usage:**

```python
from code_sandboxes import Sandbox

with Sandbox.create(variant="monty") as sandbox:
    sandbox.run_code("x = 21")
    print(sandbox.run_code("x * 2").text)  # 42

# Expose host callables and enable type checking:
from code_sandboxes.monty_sandbox import MontySandbox

sandbox = MontySandbox(
    type_check=True,
    external_functions={"now": lambda: "2026-01-01"},
)
sandbox.start()
sandbox.run_code("print(now())")
```

> Monty supports only a subset of Python — third-party libraries and rich display
> outputs are not available.

### 5. Google Colab

Runs code against a Google Colab runtime. Colab exposes a Jupyter-compatible
kernel behind an authenticating proxy, so this variant connects using
`jupyter-kernel-client`'s `ColabKernelClient`.

**Install:**

```bash
pip install code-sandboxes[colab]
```

**Parameters:**

| Parameter     | Description                           |
| ------------- | ------------------------------------- |
| `server_url`  | The Colab runtime proxy/tunnel URL    |
| `kernel_id`   | The assigned kernel identifier        |
| `proxy_token` | The `colab-runtime-proxy-token` value |

**How to obtain these values** — they are the pieces of the WebSocket URL that
Colab's own frontend uses to reach your assigned runtime:

```
wss://<host>/api/kernels/<kernel_id>/channels?session_id=<...>&colab-runtime-proxy-token=<proxy_token>&colab-client-agent=web
```

Read them from your browser's developer tools:

1. Open your notebook on [colab.research.google.com](https://colab.research.google.com)
   and **connect to a runtime** (*Runtime → Connect*, or run any cell).
1. Open DevTools (`F12`) → **Network** tab, select the **WS** filter (or type
   `kernels`), then run a cell to trigger kernel traffic.
1. Click the `.../api/kernels/<kernel_id>/channels?...` request and read off:
   - **`server_url`** — scheme + host *before* `/api/kernels` (change `wss://` to
     `https://`). Colab assigns a per-session host such as
     `https://8080-m-s-kkb-...-d.us-east1-0.prod.colab.dev`; there is usually **no**
     `/tun/m/...` path segment.
   - **`kernel_id`** — the UUID segment right after `/api/kernels/`.
   - **`proxy_token`** — the `colab-runtime-proxy-token` query parameter (same
     value as the `X-Colab-Runtime-Proxy-Token` request header). Ignore the
     `session_id` and `colab-client-agent` parameters.

The programmatic "runtime assignment API" is the internal endpoint Colab's
frontend calls (authenticated with your Google session); it is not an officially
published public API, so the DevTools method above is the practical approach. The
values are tied to your Colab session and are short-lived — refresh them after the
runtime is reassigned or reconnected.

**Usage:**

```python
from code_sandboxes import Sandbox

with Sandbox.create(
    variant="colab",
    server_url="https://8080-m-s-kkb-...-d.us-east1-0.prod.colab.dev",
    kernel_id="c9bba548-3995-4f26-8e1a-7b8fbb10c578",
    proxy_token="eyJhbGci....",
) as sandbox:
    sandbox.run_code("x = 40")
    print(sandbox.run_code("x + 2").text)  # 42
```

**Browser bridge (no manual DevTools):** instead of copying the values by hand,
set `use_browser_bridge=True` to obtain them from an authenticated Colab browser
session. This reuses `jupyter-kernel-client`'s browser bridge: a short-lived
localhost WebSocket server is opened with a one-time token, a Colab page is
launched, and the authenticated tab posts the runtime `server_url` /
`kernel_id` / `proxy_token` back to the process. Google credentials never leave
the browser.

```python
from code_sandboxes import Sandbox

with Sandbox.create(variant="colab", use_browser_bridge=True) as sandbox:
    print(sandbox.run_code("print(1 + 1)").text)
```

Install the bridge extra with `pip install 'jupyter-kernel-client[bridge]'`. The
browser side must run a cooperating page/extension/userscript that reads the
token and port from the launch URL and sends the payload (see the
`jupyter-kernel-client` README for the contract).

### 6. Modal

Runs code in a [Modal](https://modal.com/docs/guide) cloud sandbox, providing
fully isolated, on-demand containers with configurable images and secrets.

**Install:**

```bash
pip install code-sandboxes[modal]
```

**How to obtain Modal credentials:**

1. Create a free account at [modal.com](https://modal.com).

1. Authenticate the CLI — this opens a browser and writes credentials to
   `~/.modal.toml`:

   ```bash
   modal token new
   ```

   This is enough for local use: the Modal SDK reads credentials from
   `~/.modal.toml` automatically.

1. Alternatively, create a token in the Modal dashboard
   (**Settings → API Tokens**) and export it as environment variables:

   | Environment variable | Description                            |
   | -------------------- | -------------------------------------- |
   | `MODAL_TOKEN_ID`     | Modal token id (starts with `ak-`)     |
   | `MODAL_TOKEN_SECRET` | Modal token secret (starts with `as-`) |

**Do you need both `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`?**

- For environment-based auth (CI/CD, containers, hosted runners): **yes**,
  you need both values because Modal authenticates with a token pair
  (public id + secret).
- For local development with `modal token new`: **not necessarily**. The SDK can
  authenticate directly from `~/.modal.toml`.

If you need to export environment variables from your local Modal config, you can
read them from `~/.modal.toml`:

```bash
python - <<'PY'
import pathlib
import tomllib

cfg = tomllib.loads(pathlib.Path("~/.modal.toml").expanduser().read_text())
profile = cfg.get("default", cfg)
token_id = profile.get("token_id")
token_secret = profile.get("token_secret")
if token_id and token_secret:
        print(f"export MODAL_TOKEN_ID={token_id}")
        print(f"export MODAL_TOKEN_SECRET={token_secret}")
else:
        raise SystemExit("Could not find token_id/token_secret in ~/.modal.toml")
PY
```

**Parameters:** `app_name`, `image` (a prebuilt `modal.Image`), `pip_packages`
(extra packages for the default image), `python_executable`.

**Usage:**

```python
from code_sandboxes import Sandbox

with Sandbox.create(
    variant="modal",
    pip_packages=["numpy"],
) as sandbox:
    result = sandbox.run_code("import numpy as np; print(np.arange(3).sum())")
    print(result.stdout)  # "3"
```

> Each `run_code` call runs in a fresh `python -c` process, so state does not
> persist across calls. Use a single multi-statement snippet when you need shared
> state.

### 7. Datalayer

Cloud-based execution with full isolation, GPU support, snapshots, and
persistence via the [Datalayer](https://datalayer.ai) runtime, powered by the
`agent_runtimes` package.

**Install:**

```bash
pip install code-sandboxes[datalayer]
```

This extra installs `agent_runtimes`, which provides the runtime client used by
the `datalayer` sandbox variant.

**How to obtain Datalayer credentials:**

1. Create an account at [datalayer.ai](https://datalayer.ai).

1. Generate an API token from your account settings (**IAM → Tokens / API Keys**).

1. Export it (or pass it as the `token` parameter):

   | Environment variable | Description                                              |
   | -------------------- | -------------------------------------------------------- |
   | `DATALAYER_API_KEY`  | API key for Datalayer runtime authentication             |
   | `DATALAYER_RUN_URL`  | Custom Datalayer service URL (optional, for self-hosted) |

**Parameters:** `token` (defaults to `DATALAYER_API_KEY`), `run_url`,
`snapshot_name`, plus creation options like `environment`, `gpu`, `cpu`, `memory`.

**Usage:**

```python
import os
from code_sandboxes import Sandbox

os.environ["DATALAYER_API_KEY"] = "your-datalayer-token"

with Sandbox.create(
    variant="datalayer",
    gpu="A100",
    environment="python-gpu-env",
    timeout=300,
) as sandbox:
    sandbox.run_code("import torch")
    print(sandbox.run_code("print(torch.cuda.is_available())").stdout)
```

## API Reference

### Sandbox.create()

Factory method to create sandboxes:

```python
sandbox = Sandbox.create(
    variant="datalayer",  # Sandbox type
    timeout=60,                   # Execution timeout (seconds)
    environment="python-cpu-env",  # Runtime environment
    gpu="T4",                     # GPU type (T4, A100, H100, etc.)
    cpu=2.0,                      # CPU cores
    memory=4096,                  # Memory in MB
    env={"MY_VAR": "value"},      # Environment variables
    network_policy="none",        # Network access policy
    allowed_hosts=["localhost"],   # Allowlist when policy is allowlist
    tags={"project": "demo"},     # Metadata tags
)

# Network policies:
# - inherit: default behavior for the sandbox variant
# - none: block all outbound connections
# - allowlist: allow only hosts in allowed_hosts
# - all: allow all outbound connections
```

### Execution Result

```python
result = sandbox.run_code("print('hello')")

result.success    # bool: Whether execution succeeded
result.stdout     # str: Standard output
result.stderr     # str: Standard error
result.text       # str: Main result text
result.results    # list[Result]: Rich results (HTML, images, etc.)
result.code_error # CodeError: Error details if failed
result.execution_ok     # bool: Infrastructure execution status
result.execution_error  # str | None: Infrastructure error details
result.exit_code        # int | None: Exit code if code called sys.exit()

# Error handling
if not result.execution_ok:
    print(f"Sandbox failed: {result.execution_error}")
elif result.exit_code not in (None, 0):
    print(f"Process exited with code: {result.exit_code}")
elif result.code_error:
    print(f"Python error: {result.code_error.name}: {result.code_error.value}")
else:
    print(result.text)
```

### Core Methods

| Method                                   | Description                      |
| ---------------------------------------- | -------------------------------- |
| `Sandbox.create()`                       | Create a new sandbox             |
| `Sandbox.from_id(id)`                    | Reconnect to an existing sandbox |
| `Sandbox.list()`                         | List all sandboxes               |
| `sandbox.run_code(code)`                 | Execute Python code              |
| `sandbox.files.read(path)`               | Read file contents               |
| `sandbox.files.write(path, content)`     | Write file contents              |
| `sandbox.files.list(path)`               | List directory contents          |
| `sandbox.commands.run(cmd)`              | Run shell command                |
| `sandbox.commands.exec(*args)`           | Execute with streaming output    |
| `sandbox.set_timeout(seconds)`           | Update timeout                   |
| `sandbox.create_snapshot(name)`          | Save sandbox state               |
| `sandbox.terminate()` / `sandbox.kill()` | Stop sandbox                     |

## Configuration

### Environment Variables

- `DATALAYER_API_KEY`: API key for Datalayer runtime authentication

### SandboxConfig

```python
from code_sandboxes import SandboxConfig

config = SandboxConfig(
    timeout=30.0,              # Default execution timeout
    environment="python-cpu-env",
    memory_limit=4 * 1024**3,  # 4GB
    cpu_limit=2.0,
    gpu="T4",
    working_dir="/workspace",
    env_vars={"DEBUG": "1"},
    max_lifetime=3600,         # 1 hour
)

sandbox = Sandbox.create(config=config)
```

## Testing

Run the local test suite:

```bash
pytest tests/
```

Required environment variables for tests:

- None for the default local suite (`eval`, factory, model, and local Jupyter tests).

Optional environment variables for cloud-integration smoke tests:

- `DATALAYER_API_KEY`: required only when running Datalayer runtime smoke tests.
- `DATALAYER_RUN_URL`: optional custom Datalayer runtime URL.
- `DATALAYER_ENVIRONMENT`: optional environment override (for example `ai-agents-env`).
- `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`: required only if you add or run Modal integration tests.

## CI Workflows

This repository uses a reusable GitHub Actions workflow at `.github/workflows/reusable-python.yml`.

The following workflows call it:

- `.github/workflows/build.yml`
- `.github/workflows/py-tests.yml`
- `.github/workflows/py-code-style.yml`
- `.github/workflows/py-typing.yml`

Reusable workflow inputs:

- `python-version`: Python version to run.
- `install-system-deps`: Install Linux dependencies and unlock keyring.
- `install-extras`: Extras from `pyproject.toml` (for example `test,typing`).
- `extra-packages`: Additional packages installed with `uv pip install`.
- `run-tests`: Enable test execution.
- `test-command`: Command used for tests.
- `run-mypy`: Enable mypy.
- `mypy-target`: Package or module passed to mypy.
- `run-pre-commit`: Enable pre-commit checks.

## License

Copyright (c) 2025-2026 Datalayer, Inc.

BSD 3-Clause License
