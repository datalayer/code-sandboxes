# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Unit tests for Modal/Colab sandbox execution edge cases."""

from unittest.mock import MagicMock

from code_sandboxes.colab_sandbox import ColabSandbox
from code_sandboxes.modal_sandbox import ModalSandbox
from code_sandboxes.models import SandboxConfig


class _FakeStream:
    def __init__(self, text: str):
        self._text = text

    def read(self):
        return self._text


class _FakeProcess:
    def __init__(self, stdout: str, stderr: str, returncode: int):
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(stderr)
        self.returncode = returncode

    def wait(self):
        return None


def _started_modal_with_process(process: _FakeProcess) -> ModalSandbox:
    sandbox = ModalSandbox(config=SandboxConfig(timeout=10.0))
    sandbox._started = True
    sandbox._sandbox = MagicMock()
    sandbox._sandbox.exec.return_value = process
    return sandbox


def test_modal_preserves_sub_second_timeout():
    """The timeout forwarded to Modal should preserve float precision."""
    sandbox = _started_modal_with_process(_FakeProcess(stdout="", stderr="", returncode=0))

    sandbox.run_code("print('ok')", timeout=0.5)

    assert sandbox._sandbox.exec.call_args.kwargs["timeout"] == 0.5


def test_modal_code_error_does_not_set_exit_code():
    """Python exceptions should be surfaced as code_error, not exit_code."""
    sandbox = _started_modal_with_process(
        _FakeProcess(stdout="", stderr="Traceback\nValueError: boom\n", returncode=1)
    )

    result = sandbox.run_code("raise ValueError('boom')")

    assert result.code_error is not None
    assert result.code_error.name == "ValueError"
    assert result.exit_code is None


def test_modal_nonzero_return_without_stderr_sets_exit_code():
    """A non-zero return without Python traceback should set exit_code."""
    sandbox = _started_modal_with_process(_FakeProcess(stdout="", stderr="", returncode=2))

    result = sandbox.run_code("import sys; sys.exit(2)")

    assert result.code_error is None
    assert result.exit_code == 2


def test_colab_execute_exception_sets_execution_ok_false():
    """Infrastructure execute errors must set execution_ok to False."""
    sandbox = ColabSandbox(
        config=SandboxConfig(timeout=10.0),
        server_url="https://colab-host.example",
        kernel_id="kernel-id",
        proxy_token="proxy-token",  # noqa: S106
    )
    sandbox._started = True
    sandbox._client = MagicMock()
    sandbox._client.execute.side_effect = RuntimeError("connection dropped")

    result = sandbox.run_code("print('ok')")

    assert result.execution_ok is False
    assert result.execution_error is not None
    assert "Failed to execute code" in result.execution_error
