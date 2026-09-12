<!--
Copyright (c) 2025-2026 Datalayer, Inc.

BSD 3-Clause License
-->

# The protected pins' wheelhouse (PLAN_ENV.md, E1-04, E1-05)

`jupyter-server==2.21.0+datalayer.1` is Datalayer's fork, a local version no
package index carries — PyPI has no `+datalayer.1`, and no index resolves a
local version segment even if it did. Every protected pin is now forced into
every environment's own requirements (E1-04), so every resolve and every
build needs a way to satisfy this one that does not depend on the network
reaching a fork's git repository at build time.

This directory is that way: a small, `--find-links`-style wheelhouse, baked
into the base channel image at `/opt/datalayer/wheelhouse` (the contract
layer, `plane/etc/dockerfiles/jupyter-python-contract/Dockerfile`) and passed
to both `uv pip compile` (the resolver's own solve, `BuildkitResolveRunner`)
and `uv pip sync` (every builder's install, starting with the Datalayer
variant) as `--find-links /opt/datalayer/wheelhouse`. `uv` prefers an index
match when one exists and falls back to a `--find-links` directory only for
what no index has — exactly the fork, and nothing else.

## What is here, and how it was built

`jupyter_server-2.21.0+datalayer.1-py3-none-any.whl`, built from the exact
commit `services/kernels/Dockerfile` installs, `datalayer-2.21` at
`3d4873a26ba2f6871bd4f9e2cb91a7c635fe53e9` on
`datalayer-externals/jupyter-server`:

```bash
pip wheel --no-deps -w . \
  "jupyter-server @ git+https://github.com/datalayer-externals/jupyter-server@3d4873a26ba2f6871bd4f9e2cb91a7c635fe53e9#egg=jupyter_server"
```

Pure Python (`py3-none-any`), so one wheel serves every base's Python
version this channel carries. Rebuild it here, under this same file name,
whenever `services/kernels/Dockerfile`'s pinned commit changes — the
constraints file's own version pin and this wheel move together.
