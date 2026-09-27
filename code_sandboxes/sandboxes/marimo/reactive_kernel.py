# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The reactive cell graph a Marimo sandbox keeps inside its kernel.

This module *is* the code the kernel runs. :mod:`.reactive` reads this file's
text and sends it to the kernel as one execute request, so what a test
imports, a type-checker reads and the kernel executes are one and the same —
it used to be a string constant, which no tool could check.

Because its text is executed in a bare kernel namespace, the module keeps two
rules the rest of the package does not have to:

- **Self-contained.** Only the standard library at import time, and marimo —
  imported lazily, inside the methods, so the helper installs on a kernel
  that does not have marimo yet and fails with marimo's own error only when
  a cell is first registered. Never an import from ``code_sandboxes``: the
  kernel does not have it.
- **Idempotent under re-execution.** The tail binds one instance to
  ``__marimo_reactive__`` only when the name is absent, so a second driver
  executing this source on the same kernel keeps the graph the first one
  built. The same guard makes a plain ``import`` behave: the module object
  carries its own singleton.

Answers cross the protocol as one stdout line, ``__MARIMO__<base64 JSON>``,
so a caller needs nothing but the stream messages every Jupyter client
already reads.
"""

from __future__ import annotations

import base64 as _marimo_b64
import json as _marimo_json
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # only for the annotations: the kernel never imports these
    from marimo._ast.cell import CellImpl
    from marimo._runtime.dataflow import DirectedGraph
    from marimo._types.ids import CellId_t

#: What an answer line starts with on stdout.
ANSWER_MARKER = "__MARIMO__"


class _MarimoReactive:
    """A reactive cell graph, Marimo's, kept beside the kernel's namespace."""

    def __init__(self) -> None:
        from marimo._runtime.dataflow import DirectedGraph

        self._graph: DirectedGraph = DirectedGraph()
        self._code: dict[str, str] = {}

    # -- cells -----------------------------------------------------------

    def register(self, cell_id: str, code: str) -> dict[str, Any]:
        """Compile one cell and put it in the graph, replacing its old self."""
        from marimo._ast.compiler import compile_cell

        if cell_id in self._code:
            self._graph.delete_cell(cell_id)
            del self._code[cell_id]
        try:
            cell: CellImpl = compile_cell(code, cell_id=cell_id)
        except SyntaxError as error:
            return {
                "cell": cell_id,
                "error": f"SyntaxError: {error.msg} (line {error.lineno})",
            }
        self._graph.register_cell(cell_id, cell)
        self._code[cell_id] = code
        return {
            "cell": cell_id,
            "defs": sorted(cell.defs),
            "refs": sorted(cell.refs),
            "conflicts": self._conflicts(cell_id),
            "cycle": self._in_cycle(cell_id),
        }

    def remove(self, cell_id: str) -> dict[str, Any]:
        if cell_id in self._code:
            self._graph.delete_cell(cell_id)
            del self._code[cell_id]
        return {"cell": cell_id, "removed": True}

    def code(self, cell_id: str) -> str | None:
        return self._code.get(cell_id)

    def codes(self) -> dict[str, str]:
        """Every registered cell's source, by id: what a second driver on
        the same kernel reads back."""
        return dict(self._code)

    # -- what to run -----------------------------------------------------

    def plan(self, cell_id: str) -> list[str]:
        """The cells to re-run after `cell_id` ran, in dependency order."""
        from marimo._runtime.dataflow import topological_sort

        if cell_id not in self._code:
            return []
        return list(topological_sort(self._graph, list(self._graph.descendants(cell_id))))

    def plan_all(self) -> list[str]:
        """Every registered cell, in dependency order: a run-all."""
        from marimo._runtime.dataflow import topological_sort

        return list(topological_sort(self._graph, list(self._code)))

    def snapshot(self) -> dict[str, Any]:
        """The graph as data: each cell's names and neighbours, and what is wrong."""
        cells: dict[str, Any] = {}
        for cell_id in self._code:
            # marimo types its ids as `CellId_t`, a `NewType` over `str`: the
            # ids arrive as plain strings over the wire and are the same values.
            key = cast("CellId_t", cell_id)
            cell = self._graph.cells[key]
            cells[cell_id] = {
                "defs": sorted(cell.defs),
                "refs": sorted(cell.refs),
                "parents": sorted(self._graph.parents.get(key, set())),
                "children": sorted(self._graph.children.get(key, set())),
            }
        return {
            "cells": cells,
            "conflicts": sorted(self._graph.get_multiply_defined()),
            "cycles": sorted(self._cycle_cells()),
        }

    # -- diagnostics -----------------------------------------------------

    def _conflicts(self, cell_id: str) -> list[str]:
        defs = self._graph.cells[cast("CellId_t", cell_id)].defs
        return sorted(name for name in self._graph.get_multiply_defined() if name in defs)

    def _cycle_cells(self) -> set[str]:
        cells: set[str] = set()
        for cycle in self._graph.cycles:
            for edge in cycle:
                if isinstance(edge, (tuple, list)):
                    cells.update(edge)
                else:
                    cells.add(edge)
        return cells

    def _in_cycle(self, cell_id: str) -> bool:
        return cell_id in self._cycle_cells()

    # -- the wire --------------------------------------------------------

    def answer(self, method: str, *args: Any) -> None:
        """Print one method's result as a marked base64 JSON line."""
        payload = _marimo_json.dumps(getattr(self, method)(*args))
        print(ANSWER_MARKER + _marimo_b64.b64encode(payload.encode("utf-8")).decode("ascii"))


if "__marimo_reactive__" not in globals():
    __marimo_reactive__ = _MarimoReactive()
