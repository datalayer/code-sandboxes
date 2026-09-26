# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Marimo's reactive cells over any sandbox that speaks the Jupyter protocol.

`MarimoSandbox` is one variant: a Jupyter *server* sandbox with the graph in
its kernel. A sandbox somewhere else — a Datalayer runtime, a Kaggle kernel,
anything a `CodeSandboxClient` wraps — is a kernel too, and the graph needs
nothing from a kernel but execute requests and stdout (`marimo_reactive`). So
this is the same reactivity as a *driver* over a client rather than as a
variant: install the helper into whatever kernel the client reaches, then
run cells through the client's Jupyter-shaped `execute`, re-running the
dependents the graph names.

    cells = MarimoCells(client)          # any CodeSandboxClient
    cells.run_cell("a", "x = 1")
    run = cells.run_cell("b", "print(x)")
    cells.run_cell("a", "x = 2").reactions[0].cell_id    # "b", re-run

This is what an MCP toolset uses to give an agent reactive cells on the
sandbox its session already holds (code-sandboxes#37). On a client whose
sandbox is a `MarimoSandbox`, the sandbox's own graph is used — the same cells
seen from either side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from .marimo_reactive import (
    HELPER_NAME,
    KERNEL_HELPER_SOURCE,
    decode_answer,
    question,
)

logger = logging.getLogger(__name__)

#: How the helper's answer is found in a reply: the stdout stream outputs.
_STDOUT = ("stream", "stdout")


class JupyterShaped(Protocol):
    """What the driver needs of a client: `execute` answering a Jupyter-shaped reply."""

    def execute(self, code: str, timeout: float | None = None, **kwargs: Any) -> dict[str, Any]: ...


@dataclass
class CellReply:
    """One cell's execution, as the Jupyter-shaped reply the client answered."""

    cell_id: str
    code: str
    reply: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.reply.get("status", "ok") == "ok"

    @property
    def outputs(self) -> list[dict[str, Any]]:
        return list(self.reply.get("outputs", []))


@dataclass
class CellsRun:
    """What running one cell did: the cell, then what it made re-run."""

    cell: CellReply
    reactions: list[CellReply] = field(default_factory=list)
    #: The cell's names as the graph read them, or its `error` when the source
    #: does not parse (the cell then never ran).
    registration: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return "error" not in self.registration and self.cell.ok

    def to_dict(self) -> dict[str, Any]:
        """The run as plain data, the shape a tool answers with."""
        return {
            "cell_id": self.cell.cell_id,
            "status": "error" if not self.ok else "ok",
            "outputs": self.cell.outputs,
            "registration": self.registration,
            "reactions": [
                {
                    "cell_id": reaction.cell_id,
                    "code": reaction.code,
                    "status": "ok" if reaction.ok else "error",
                    "outputs": reaction.outputs,
                }
                for reaction in self.reactions
            ],
        }


class MarimoCells:
    """Marimo's reactive cell graph, driven through a client's `execute`.

    ``install_marimo`` says whether a kernel that lacks marimo gets it
    installed with pip (once, at `install`); ``False`` refuses instead.
    """

    def __init__(
        self,
        client: JupyterShaped,
        *,
        install_marimo: bool = True,
        timeout: float | None = None,
    ):
        self._client = client
        self._install_marimo = install_marimo
        self._timeout = timeout
        self._ready = False
        self._cells: dict[str, str] = {}
        self._anonymous = 0

    # -- the kernel ------------------------------------------------------

    def _execute(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        return self._client.execute(code, timeout=timeout if timeout is not None else self._timeout)

    @staticmethod
    def _stdout_lines(reply: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        for output in reply.get("outputs", []):
            if (output.get("output_type"), output.get("name")) == _STDOUT:
                lines.extend(str(output.get("text", "")).splitlines())
        return lines

    @staticmethod
    def _error_of(reply: dict[str, Any]) -> str | None:
        for output in reply.get("outputs", []):
            if output.get("output_type") == "error":
                return f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
        return None if reply.get("status", "ok") == "ok" else "the execution failed"

    def install(self) -> None:
        """Put the graph in the kernel, installing marimo first if it is missing.

        Idempotent: the helper keeps the graph it already holds, so a driver
        created again on the same sandbox — another replica answering the
        session, say — sees the cells registered before.
        """
        if self._ready:
            return
        probe = self._execute("import marimo")
        if self._error_of(probe) is not None:
            if not self._install_marimo:
                raise RuntimeError(
                    "The kernel has no marimo and install_marimo is False: "
                    "install marimo in the environment first."
                )
            logger.info("Installing marimo into the kernel.")
            installed = self._execute(
                "import subprocess, sys\n"
                "subprocess.check_call("
                "[sys.executable, '-m', 'pip', 'install', '--quiet', 'marimo'])"
            )
            problem = self._error_of(installed)
            if problem is not None:
                raise RuntimeError(f"marimo could not be installed: {problem}")
        bootstrap = self._execute(KERNEL_HELPER_SOURCE)
        problem = self._error_of(bootstrap)
        if problem is not None:
            raise RuntimeError(f"The Marimo reactive helper could not start: {problem}")
        self._ready = True
        # Cells registered before this driver existed (the same kernel, another
        # driver): read them back so `cells` and the reactions know their code.
        for cell_id, code in self._ask("codes").items():
            self._cells.setdefault(cell_id, code)

    def _ask(self, method: str, *args: Any) -> Any:
        self.install()
        reply = self._execute(question(method, *args))
        problem = self._error_of(reply)
        if problem is not None:
            raise RuntimeError(f"The Marimo graph refused {method}: {problem}")
        return decode_answer(self._stdout_lines(reply))

    # -- the graph -------------------------------------------------------

    def register_cell(self, cell_id: str, code: str) -> dict[str, Any]:
        """Put a cell in the graph without running it; answers its names."""
        answer = self._ask("register", cell_id, code)
        if "error" in answer:
            self._cells.pop(cell_id, None)
        else:
            self._cells[cell_id] = code
        return answer

    def remove_cell(self, cell_id: str) -> None:
        self._cells.pop(cell_id, None)
        self._ask("remove", cell_id)

    def plan(self, cell_id: str) -> list[str]:
        """The cells that re-run after `cell_id`, in dependency order."""
        return [str(dep) for dep in self._ask("plan", cell_id)]

    def graph(self) -> dict[str, Any]:
        """Every cell's names and neighbours, the conflicts and the cycles."""
        return self._ask("snapshot")

    @property
    def cells(self) -> dict[str, str]:
        """The registered cells' source, by id — the kernel's, read back first."""
        self.install()
        return dict(self._cells)

    # -- running ---------------------------------------------------------

    def run_cell(
        self, cell_id: str, code: str, *, react: bool = True, timeout: float | None = None
    ) -> CellsRun:
        """Run one cell, then the cells that depend on it.

        A cell that fails stops the reaction: nothing downstream runs on a
        state the failure left half-made, which is what Marimo does too.
        """
        registration = self.register_cell(cell_id, code)
        if "error" in registration:
            reply = {
                "status": "error",
                "outputs": [
                    {
                        "output_type": "error",
                        "ename": "SyntaxError",
                        "evalue": registration["error"],
                        "traceback": [],
                    }
                ],
            }
            return CellsRun(cell=CellReply(cell_id, code, reply), registration=registration)
        run = CellsRun(
            cell=CellReply(cell_id, code, self._execute(code, timeout)),
            registration=registration,
        )
        if not react or not run.ok:
            return run
        for dependent in self.plan(cell_id):
            dependent_code = self._cells.get(dependent)
            if dependent_code is None:
                continue
            reaction = CellReply(dependent, dependent_code, self._execute(dependent_code, timeout))
            run.reactions.append(reaction)
            if not reaction.ok:
                break
        return run

    def run_code(self, code: str, *, timeout: float | None = None) -> CellsRun:
        """Run code as a cell of its own, reactively."""
        self._anonymous += 1
        return self.run_cell(f"cell-{self._anonymous}", code, timeout=timeout)

    def __repr__(self) -> str:
        return f"MarimoCells(cells={len(self._cells)}, helper={HELPER_NAME!r})"
