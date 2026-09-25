# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""A Marimo sandbox: a Jupyter kernel with Marimo's reactivity in it.

Marimo notebooks are reactive — running a cell re-runs every cell that depends
on what it defined — and Marimo works that out from the cells' source, not from
running them (code-sandboxes#34). So a Marimo sandbox is a Jupyter server
sandbox whose kernel also holds Marimo's dataflow graph (`marimo_reactive`):
the client API stays `jupyter-kernel-client` and the Jupyter protocol, and the
graph is driven through ordinary execute requests.

    with Sandbox.create(variant="marimo") as sandbox:
        sandbox.run_cell("a", "x = 1")
        sandbox.run_cell("b", "y = x + 1\\nprint(y)")   # prints 2
        run = sandbox.run_cell("a", "x = 10")            # re-runs b: prints 11
        run.reactions[0].cell_id                          # "b"

`run_code` keeps working as on every sandbox — each call is a cell of its own
— so an agent that never heard of cells still gets a consistent state: what it
ran earlier and that depends on what it just ran is run again.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .jupyter_server_sandbox import JupyterServerSandbox
from .marimo_reactive import (
    HELPER_NAME,
    KERNEL_HELPER_SOURCE,
    decode_answer,
    question,
)
from .models import (
    CodeError,
    Context,
    ExecutionResult,
    OutputHandler,
    OutputMessage,
    Result,
    SandboxEnvironment,
)

logger = logging.getLogger(__name__)

VARIANT = "marimo"


@dataclass
class CellRun:
    """One cell's execution, by the cell's id."""

    cell_id: str
    code: str
    result: ExecutionResult


@dataclass
class MarimoRun:
    """What running one cell did: the cell, then what it made re-run."""

    cell_id: str
    result: ExecutionResult
    #: Cells re-run because they depend on this one, in the order they ran.
    reactions: list[CellRun] = field(default_factory=list)
    #: The cell's names, as the graph read them (`defs`, `refs`, `conflicts`,
    #: `cycle`), or its `error` when the source does not parse.
    registration: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.result.execution_ok and self.result.code_error is None


class MarimoSandbox(JupyterServerSandbox):
    """Jupyter server sandbox with Marimo's reactive cell graph in the kernel.

    Pass ``install_marimo=False`` to refuse to install marimo into a kernel
    that lacks it (the default installs it with pip, once, at start).
    """

    def __init__(self, *args: Any, install_marimo: bool = True, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._install_marimo = install_marimo
        self._helper_ready = False
        self._cells: dict[str, str] = {}
        self._anonymous = 0

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        super().start()
        if self._info is not None:
            self._info.variant = VARIANT
            self._info.metadata = {**(self._info.metadata or {}), "reactive": "marimo"}
        self._install_helper()

    @classmethod
    def list_environments(cls) -> list[SandboxEnvironment]:
        return [
            SandboxEnvironment(
                name=VARIANT,
                title="Marimo",
                language="python",
                owner="local",
                visibility="local",
                burning_rate=0.0,
                metadata={"variant": VARIANT, "reactive": "marimo"},
            )
        ]

    def _install_helper(self) -> None:
        """Put the graph in the kernel, installing marimo first if it is missing."""
        probe = self._plain_run("import marimo")
        if probe.code_error is not None:
            if not self._install_marimo:
                raise RuntimeError(
                    "The kernel has no marimo and install_marimo is False: "
                    "install marimo in the environment first."
                )
            logger.info("Installing marimo into the kernel.")
            installed = self._plain_run(
                "import subprocess, sys\n"
                "subprocess.check_call("
                "[sys.executable, '-m', 'pip', 'install', '--quiet', 'marimo'])"
            )
            if installed.code_error is not None:
                raise RuntimeError(f"marimo could not be installed: {installed.code_error.value}")
        bootstrap = self._plain_run(KERNEL_HELPER_SOURCE)
        if bootstrap.code_error is not None:
            raise RuntimeError(
                f"The Marimo reactive helper could not start: {bootstrap.code_error.value}"
            )
        self._helper_ready = True

    def _plain_run(self, code: str, **handlers: Any) -> ExecutionResult:
        """`run_code` as the parent does it: no cell, no reactivity."""
        return JupyterServerSandbox.run_code(self, code, **handlers)

    def _ask(self, method: str, *args: Any) -> Any:
        if not self._helper_ready:
            raise RuntimeError("The Marimo sandbox is not started.")
        result = self._plain_run(question(method, *args))
        if result.code_error is not None:
            error = result.code_error
            raise RuntimeError(f"The Marimo graph refused {method}: {error.name}: {error.value}")
        return decode_answer([message.line for message in result.logs.stdout])

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
        """The registered cells' source, by id."""
        return dict(self._cells)

    # -- running ---------------------------------------------------------

    def run_cell(
        self,
        cell_id: str,
        code: str,
        *,
        react: bool = True,
        on_stdout: OutputHandler[OutputMessage] | None = None,
        on_stderr: OutputHandler[OutputMessage] | None = None,
        on_result: OutputHandler[Result] | None = None,
        on_error: OutputHandler[CodeError] | None = None,
        timeout: float | None = None,
    ) -> MarimoRun:
        """Run one cell, then the cells that depend on it.

        A cell that fails stops the reaction: nothing downstream runs on a
        state the failure left half-made, which is what Marimo does too.
        """
        handlers = {
            "on_stdout": on_stdout,
            "on_stderr": on_stderr,
            "on_result": on_result,
            "on_error": on_error,
        }
        registration = self.register_cell(cell_id, code)
        if "error" in registration:
            return MarimoRun(
                cell_id=cell_id,
                result=ExecutionResult(
                    execution_ok=True,
                    code_error=CodeError(
                        name="SyntaxError", value=registration["error"], traceback=""
                    ),
                    started_at=time.time(),
                    completed_at=time.time(),
                    context_id="default",
                ),
                registration=registration,
            )
        result = self._plain_run(code, timeout=timeout, **handlers)
        run = MarimoRun(cell_id=cell_id, result=result, registration=registration)
        if not react or not run.ok:
            return run
        for dependent in self.plan(cell_id):
            dependent_code = self._cells.get(dependent)
            if dependent_code is None:
                continue
            dependent_result = self._plain_run(dependent_code, timeout=timeout, **handlers)
            run.reactions.append(
                CellRun(cell_id=dependent, code=dependent_code, result=dependent_result)
            )
            if dependent_result.code_error is not None:
                break
        return run

    def run_code(  # type: ignore[override]
        self,
        code: str,
        language: str = "python",
        context: Context | None = None,
        on_stdout: OutputHandler[OutputMessage] | None = None,
        on_stderr: OutputHandler[OutputMessage] | None = None,
        on_result: OutputHandler[Result] | None = None,
        on_error: OutputHandler[CodeError] | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Run code as a cell of its own, reactively; answers that cell's result.

        The cells it made re-run are on the result's ``marimo_reactions``
        (cell ids), so a caller that wants them can ask `cells` for their code.
        """
        if language != "python":
            raise ValueError(f"MarimoSandbox only supports Python, got: {language}")
        if envs:
            env_code = "\n".join(f"import os; os.environ[{k!r}] = {v!r}" for k, v in envs.items())
            self._plain_run(env_code)
        self._anonymous += 1
        run = self.run_cell(
            f"cell-{self._anonymous}",
            code,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            on_result=on_result,
            on_error=on_error,
            timeout=timeout,
        )
        result = run.result
        if run.reactions:
            # `ExecutionResult` allows extra fields; the ids are enough to
            # find the cells again, and their results are the reactions'.
            try:
                result.marimo_reactions = [reaction.cell_id for reaction in run.reactions]  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                pass
        return result

    def __repr__(self) -> str:
        return f"MarimoSandbox(cells={len(self._cells)}, helper={HELPER_NAME!r})"
