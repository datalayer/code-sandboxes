# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The 1.9.x flat module paths still resolve, to the very same modules.

Released consumers import them (jupyter-mcp-sandboxes 0.2.6:
`code_sandboxes.datalayer_sandbox`), so 1.10.0 keeps each old path as an
alias of its new home until 2.0.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest

#: Every 1.9.x module path, written out rather than read from the package:
#: a path dropped from the alias table, or one typed wrong in it, fails here
#: instead of passing a test built from the same table.
PAIRS = [
    ("cloudflare_sandbox", "sandboxes.cloudflare.cloudflare"),
    ("coreweave_sandbox", "sandboxes.coreweave.coreweave"),
    ("datalayer_sandbox", "sandboxes.datalayer.datalayer"),
    ("daytona_sandbox", "sandboxes.daytona.daytona"),
    ("docker_sandbox", "sandboxes.docker.docker"),
    ("e2b_sandbox", "sandboxes.e2b.e2b"),
    ("eval_sandbox", "sandboxes.eval.eval"),
    ("google_colab", "sandboxes.google_colab.client"),
    ("google_colab_sandbox", "sandboxes.google_colab.google_colab"),
    ("jupyter_server_sandbox", "sandboxes.jupyter_server.jupyter_server"),
    ("kaggle", "sandboxes.kaggle.client"),
    ("kaggle_execute", "sandboxes.kaggle.execute"),
    ("kaggle_live", "sandboxes.kaggle.live"),
    ("kaggle_sandbox", "sandboxes.kaggle.kaggle"),
    ("marimo_cells", "sandboxes.marimo.cells"),
    ("marimo_reactive", "sandboxes.marimo.reactive"),
    ("marimo_sandbox", "sandboxes.marimo.marimo"),
    ("modal_sandbox", "sandboxes.modal.modal"),
    ("monty_sandbox", "sandboxes.monty.monty"),
]


def test_the_alias_table_is_exactly_the_nineteen_flat_paths():
    import code_sandboxes

    assert code_sandboxes._FLAT_PATHS == dict(PAIRS)


@pytest.mark.parametrize("old,new", PAIRS)
def test_the_old_path_is_the_new_module(old, new):
    assert importlib.import_module(f"code_sandboxes.{old}") is importlib.import_module(
        f"code_sandboxes.{new}"
    )


def test_from_import_on_the_old_path():
    from code_sandboxes.datalayer_sandbox import DatalayerSandbox
    from code_sandboxes.sandboxes.datalayer import DatalayerSandbox as Moved

    assert DatalayerSandbox is Moved


def test_patching_through_the_old_path_patches_the_real_class():
    """What jupyter-mcp-sandboxes' tests do."""
    from code_sandboxes.sandboxes.datalayer import DatalayerSandbox

    with patch("code_sandboxes.datalayer_sandbox.DatalayerSandbox.from_id", return_value="x"):
        assert DatalayerSandbox.from_id("anything") == "x"
