# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The reactive cell graph a Marimo sandbox keeps inside its kernel.

Marimo's reactivity is a dataflow graph: a cell *defines* names and *refers*
to names, and running a cell re-runs every cell that, transitively, refers to
what it defined. Marimo computes that graph statically from the source
(`marimo._ast.compiler`) and orders the re-runs topologically
(`marimo._runtime.dataflow`). Nothing about it needs Marimo's own server: the
graph can live in an ordinary IPython kernel and be driven over the Jupyter
protocol, which is what code-sandboxes#34 asks for and what keeps
`jupyter-kernel-client` the client API.

The helper below is the source that is sent to the kernel, as one execute
request, the first time a Marimo sandbox needs it. It is also executed here at
import time, so the tests exercise exactly the code the kernel runs and this
package can answer graph questions without a kernel. The same source, verbatim,
is carried by `@datalayer/jupyter-react` (`jupyter/marimo/reactive.ts`) for the
browser-side components; keep the two copies identical.

Answers cross the protocol as one stdout line, `__MARIMO__<base64 JSON>`, so a
caller needs nothing but the stream messages every Jupyter client already
reads.
"""

from __future__ import annotations

from typing import Any

#: The name the helper is bound to in the kernel's user namespace.
HELPER_NAME = "__marimo_reactive__"

#: What an answer line starts with on stdout.
ANSWER_MARKER = "__MARIMO__"

KERNEL_HELPER_SOURCE = r'''
import base64 as _marimo_b64
import json as _marimo_json


class _MarimoReactive:
    """A reactive cell graph, Marimo's, kept beside the kernel's namespace."""

    def __init__(self):
        from marimo._runtime.dataflow import DirectedGraph

        self._graph = DirectedGraph()
        self._code = {}

    # -- cells -----------------------------------------------------------

    def register(self, cell_id, code):
        """Compile one cell and put it in the graph, replacing its old self."""
        from marimo._ast.compiler import compile_cell

        if cell_id in self._code:
            self._graph.delete_cell(cell_id)
            del self._code[cell_id]
        try:
            cell = compile_cell(code, cell_id=cell_id)
        except SyntaxError as error:
            return {
                "cell": cell_id,
                "error": "SyntaxError: %s (line %s)" % (error.msg, error.lineno),
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

    def remove(self, cell_id):
        if cell_id in self._code:
            self._graph.delete_cell(cell_id)
            del self._code[cell_id]
        return {"cell": cell_id, "removed": True}

    def code(self, cell_id):
        return self._code.get(cell_id)

    # -- what to run -----------------------------------------------------

    def plan(self, cell_id):
        """The cells to re-run after `cell_id` ran, in dependency order."""
        from marimo._runtime.dataflow import topological_sort

        if cell_id not in self._code:
            return []
        return list(topological_sort(self._graph, list(self._graph.descendants(cell_id))))

    def plan_all(self):
        """Every registered cell, in dependency order: a run-all."""
        from marimo._runtime.dataflow import topological_sort

        return list(topological_sort(self._graph, list(self._code)))

    def snapshot(self):
        """The graph as data: each cell's names and neighbours, and what is wrong."""
        cells = {}
        for cell_id in self._code:
            cell = self._graph.cells[cell_id]
            cells[cell_id] = {
                "defs": sorted(cell.defs),
                "refs": sorted(cell.refs),
                "parents": sorted(self._graph.parents.get(cell_id, ())),
                "children": sorted(self._graph.children.get(cell_id, ())),
            }
        return {
            "cells": cells,
            "conflicts": sorted(self._graph.get_multiply_defined()),
            "cycles": sorted(self._cycle_cells()),
        }

    # -- diagnostics -----------------------------------------------------

    def _conflicts(self, cell_id):
        defs = self._graph.cells[cell_id].defs
        return sorted(name for name in self._graph.get_multiply_defined() if name in defs)

    def _cycle_cells(self):
        cells = set()
        for cycle in self._graph.cycles:
            for edge in cycle:
                if isinstance(edge, (tuple, list)):
                    cells.update(edge)
                else:
                    cells.add(edge)
        return cells

    def _in_cycle(self, cell_id):
        return cell_id in self._cycle_cells()

    # -- the wire --------------------------------------------------------

    def answer(self, method, *args):
        """Print one method's result as a marked base64 JSON line."""
        payload = _marimo_json.dumps(getattr(self, method)(*args))
        print("__MARIMO__" + _marimo_b64.b64encode(payload.encode("utf-8")).decode("ascii"))


if "__marimo_reactive__" not in globals():
    __marimo_reactive__ = _MarimoReactive()
'''


def decode_answer(stdout_lines: list[str]) -> Any:
    """The helper's answer, from the stdout lines an execution produced.

    Raises `LookupError` when no answer line is there: the helper is not
    installed, or the kernel wrote an error instead.
    """
    import base64
    import json

    for line in reversed(stdout_lines):
        line = line.strip()
        if line.startswith(ANSWER_MARKER):
            return json.loads(base64.b64decode(line[len(ANSWER_MARKER) :]).decode("utf-8"))
    raise LookupError("The kernel returned no Marimo graph answer.")


def question(method: str, *args: Any) -> str:
    """The code that asks the kernel's helper one thing."""
    arguments = ", ".join(repr(argument) for argument in args)
    return f"{HELPER_NAME}.answer({method!r}{', ' if arguments else ''}{arguments})"


def local_helper() -> Any:
    """The helper, executed here: the graph without a kernel (needs marimo)."""
    namespace: dict[str, Any] = {}
    exec(KERNEL_HELPER_SOURCE, namespace)  # noqa: S102 - our own source, above
    return namespace[HELPER_NAME]
