<!--
  ~ Copyright (c) 2025-2026 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

[![Datalayer](https://assets.datalayer.tech/datalayer-25.svg)](https://datalayer.io)

[![Become a Sponsor](https://img.shields.io/static/v1?label=Become%20a%20Sponsor&message=%E2%9D%A4&logo=GitHub&style=flat&color=1ABC9C)](https://github.com/sponsors/datalayer)

# { } 📦 Code Sandboxes

[![PyPI - Version](https://img.shields.io/pypi/v/code-sandboxes)](https://pypi.org/project/code-sandboxes)

Code Sandboxes (`code_sandboxes`) is a Python package for running code in isolated sandbox variants through a unified API.

Canonical variant names:

- `jupyter`
- `docker`
- `eval`
- `monty`
- `kaggle`
- `colab`
- `modal`
- `datalayer`

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

## Quick Examples

### Python: launch a `jupyter` sandbox

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

### CLI REPL: `kaggle` variant

Kaggle REPL supports both interactive runtime mode and credential-based batch mode.

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

For full setup and parameters for all variants, see:

- [https://code-sandboxes.datalayer.tech/sandboxes](https://code-sandboxes.datalayer.tech/sandboxes)
- [https://code-sandboxes.datalayer.tech/cli](https://code-sandboxes.datalayer.tech/cli)
- [https://code-sandboxes.datalayer.tech/installation](https://code-sandboxes.datalayer.tech/installation)

## License

BSD 3-Clause License
