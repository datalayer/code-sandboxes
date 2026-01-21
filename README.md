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

Four variants are available:

| Variant | Isolation | Use Case |
|---------|-----------|----------|
| `local-eval` | None (Python exec) | Development, testing |
| `local-docker` | Container (Jupyter Server) | Local isolated execution |
| `local-jupyter` | Process (Jupyter kernel) | Local persistent state |
| `datalayer-runtime` | Cloud VM | Production, GPU workloads |

## Installation

```bash
# Basic installation
pip install code-sandboxes

# With Datalayer runtime support
pip install code-sandboxes[datalayer]

# With Docker support
pip install code-sandboxes[docker]

# All features
pip install code-sandboxes[all]
```

### Docker Variant Setup

The `local-docker` variant runs a Jupyter Server inside a Docker container and uses
`jupyter-kernel-client` to execute code.

Build the Docker image used by `LocalDockerSandbox`:

```bash
docker build -t code-sandboxes-jupyter:latest -f docker/Dockerfile .
```

## Quick Start

### Simple Code Execution

```python
from code_sandboxes import Sandbox

# Create a sandbox with timeout
with Sandbox.create(variant="local-eval", timeout=60) as sandbox:
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
    variant="datalayer-runtime",
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
with Sandbox.create(variant="datalayer-runtime") as sandbox:
    # Set up environment
    sandbox.install_packages(["pandas", "numpy"])
    sandbox.run_code("import pandas as pd; df = pd.DataFrame({'a': [1,2,3]})")
    
    # Create snapshot
    snapshot = sandbox.create_snapshot("my-setup")
    print(f"Snapshot created: {snapshot.id}")

# Later: restore from snapshot
with Sandbox.create(variant="datalayer-runtime", snapshot_name="my-setup") as sandbox:
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

## API Reference

### Sandbox.create()

Factory method to create sandboxes:

```python
sandbox = Sandbox.create(
    variant="datalayer-runtime",  # Sandbox type
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
result.error      # ExecutionError: Error details if failed
```

### Core Methods

| Method | Description |
|--------|-------------|
| `Sandbox.create()` | Create a new sandbox |
| `Sandbox.from_id(id)` | Reconnect to an existing sandbox |
| `Sandbox.list()` | List all sandboxes |
| `sandbox.run_code(code)` | Execute Python code |
| `sandbox.files.read(path)` | Read file contents |
| `sandbox.files.write(path, content)` | Write file contents |
| `sandbox.files.list(path)` | List directory contents |
| `sandbox.commands.run(cmd)` | Run shell command |
| `sandbox.commands.exec(*args)` | Execute with streaming output |
| `sandbox.set_timeout(seconds)` | Update timeout |
| `sandbox.create_snapshot(name)` | Save sandbox state |
| `sandbox.terminate()` / `sandbox.kill()` | Stop sandbox |

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

## License

Copyright (c) 2025-2026 Datalayer, Inc.

BSD 3-Clause License
