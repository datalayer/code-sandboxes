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

The helper is real code now: :mod:`.reactive_kernel`, a typed module of its
own, whose *file text* is what `KERNEL_HELPER_SOURCE` holds and what a sandbox
sends to the kernel as one execute request the first time it needs the graph.
Importing that module, executing its text (`local_helper`) and running it in a
kernel are the same code, so the type-checker and the tests read exactly what
the kernel runs. The same source, verbatim, is carried by
`@datalayer/jupyter-react` (`jupyter/marimo/reactive.ts`) for the browser-side
components; keep the two copies identical.

Answers cross the protocol as one stdout line, `__MARIMO__<base64 JSON>`, so a
caller needs nothing but the stream messages every Jupyter client already
reads.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

#: The name the helper is bound to in the kernel's user namespace.
HELPER_NAME = "__marimo_reactive__"

#: What an answer line starts with on stdout.
ANSWER_MARKER = "__MARIMO__"

#: The code the kernel runs: `reactive_kernel.py`'s own text, read from the
#: installed package rather than written out twice. Reading it — instead of
#: importing the module here — keeps `code_sandboxes` importable in an
#: environment without marimo; the module's tail builds the graph, which
#: imports marimo.
KERNEL_HELPER_SOURCE = (
    resources.files(__package__).joinpath("reactive_kernel.py").read_text(encoding="utf-8")
)


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
