# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Unit tests for Modal/Colab sandbox execution edge cases."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from code_sandboxes.colab_sandbox import ColabSandbox
from code_sandboxes.kaggle_sandbox import KaggleSandbox
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


class _FakeModalRuntime:
    def __init__(self, process: _FakeProcess):
        self._process = process
        self.exec_kwargs: dict | None = None

    def exec(self, *_args, **kwargs):
        self.exec_kwargs = kwargs
        return self._process


def _started_modal_with_process(process: _FakeProcess) -> ModalSandbox:
    sandbox = ModalSandbox(config=SandboxConfig(timeout=10.0))
    sandbox._started = True
    sandbox._sandbox = _FakeModalRuntime(process)
    return sandbox


def test_modal_sub_second_timeout_is_rounded_for_modal_exec():
    """Modal exec expects integer seconds, so sub-second values are rounded up."""
    sandbox = _started_modal_with_process(_FakeProcess(stdout="", stderr="", returncode=0))

    sandbox.run_code("print('ok')", timeout=0.5)

    assert sandbox._sandbox.exec_kwargs is not None
    assert sandbox._sandbox.exec_kwargs["timeout"] == 1


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
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


def test_kaggle_execute_exception_sets_execution_ok_false():
    """Infrastructure execute errors must set execution_ok to False."""
    sandbox = KaggleSandbox(
        config=SandboxConfig(timeout=10.0),
        server_url="https://kaggle-host.example/proxy",
        kernel_id="kernel-id",
    )
    sandbox._started = True
    sandbox._client = MagicMock()
    sandbox._client.execute.side_effect = RuntimeError("connection dropped")

    result = sandbox.run_code("print('ok')")

    assert result.execution_ok is False
    assert result.execution_error is not None
    assert "Failed to execute code" in result.execution_error


def test_kaggle_batch_mode_runs_without_runtime_connection(monkeypatch):
    """Without runtime URL/channels, KaggleSandbox falls back to batch executor."""

    class _FakeKaggleExecutor:
        def __init__(self, username=None, quiet=True):
            self.username = username
            self.quiet = quiet

        def execute(self, code, wait=True, timeout=0.0, download_output=True, accelerator=None):
            assert "print('ok')" in code
            return SimpleNamespace(
                slug="demo-slug",
                status="complete",
                url="https://www.kaggle.com/code/demo/demo-slug",
                version_number=1,
                failure_message=None,
                output_dir=None,
                output_files=[],
                log="ok\n42",
                notebook=None,
                succeeded=True,
            )

    monkeypatch.setattr(
        "code_sandboxes.kaggle_sandbox.KaggleKernelExecutor",
        _FakeKaggleExecutor,
    )

    sandbox = KaggleSandbox(config=SandboxConfig(timeout=10.0))
    sandbox.start()

    result = sandbox.run_code("print('ok')")

    assert result.execution_ok is True
    assert result.code_error is None
    assert "ok" in result.stdout
    assert result.text == "ok\n42"
    assert sandbox.info is not None
    assert sandbox.info.metadata["mode"] == "batch"

    sandbox.stop()


def test_kaggle_batch_mode_maps_job_failure_to_code_error(monkeypatch):
    """A failed Kaggle batch job is returned as a code-level execution error."""

    class _FakeKaggleExecutor:
        def __init__(self, username=None, quiet=True):
            self.username = username
            self.quiet = quiet

        def execute(self, code, wait=True, timeout=0.0, download_output=True, accelerator=None):
            return SimpleNamespace(
                slug="demo-slug",
                status="error",
                url="https://www.kaggle.com/code/demo/demo-slug",
                version_number=1,
                failure_message="Notebook failed",
                output_dir=None,
                output_files=[],
                log="Traceback\nValueError: boom",
                notebook=None,
                succeeded=False,
            )

    monkeypatch.setattr(
        "code_sandboxes.kaggle_sandbox.KaggleKernelExecutor",
        _FakeKaggleExecutor,
    )

    sandbox = KaggleSandbox(config=SandboxConfig(timeout=10.0))
    sandbox.start()

    result = sandbox.run_code("raise ValueError('boom')")

    assert result.execution_ok is True
    assert result.code_error is not None
    assert result.code_error.name == "KaggleExecutionError"
    assert result.stderr == "Notebook failed"

    sandbox.stop()


def test_kaggle_batch_mode_forwards_gpu_as_accelerator(monkeypatch):
    """Kaggle batch mode should map sandbox gpu setting to executor accelerator."""

    captured: dict[str, str | None] = {}

    class _FakeKaggleExecutor:
        def __init__(self, username=None, quiet=True):
            self.username = username
            self.quiet = quiet

        def execute(self, code, wait=True, timeout=0.0, download_output=True, accelerator=None):
            captured["accelerator"] = accelerator
            return SimpleNamespace(
                slug="demo-slug",
                status="complete",
                url="https://www.kaggle.com/code/demo/demo-slug",
                version_number=1,
                failure_message=None,
                output_dir=None,
                output_files=[],
                log="ok",
                notebook=None,
                succeeded=True,
            )

    monkeypatch.setattr(
        "code_sandboxes.kaggle_sandbox.KaggleKernelExecutor",
        _FakeKaggleExecutor,
    )

    sandbox = KaggleSandbox(config=SandboxConfig(timeout=10.0, gpu="T4"))
    sandbox.start()
    sandbox.run_code("print('ok')")

    assert captured["accelerator"] == "T4"

    sandbox.stop()


def test_kaggle_batch_mode_consumes_kernel_like_reply(monkeypatch):
    """Batch mode should map kernel-like reply to logs/results like interactive mode."""

    class _FakeKaggleResult:
        slug = "demo-slug"
        status = "COMPLETE"
        url = "https://www.kaggle.com/code/demo/demo-slug"
        version_number = 1
        failure_message = None
        output_dir = None
        output_files: ClassVar[list[str]] = []
        log = None
        succeeded = True

        @staticmethod
        def to_kernel_reply():
            return {
                "execution_count": 7,
                "status": "ok",
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": "hello from kaggle\\n"},
                    {
                        "output_type": "execute_result",
                        "data": {"text/plain": "42"},
                        "metadata": {},
                    },
                ],
            }

    class _FakeKaggleExecutor:
        def __init__(self, username=None, quiet=True):
            self.username = username
            self.quiet = quiet

        def execute(self, code, wait=True, timeout=0.0, download_output=True, accelerator=None):
            return _FakeKaggleResult()

    monkeypatch.setattr(
        "code_sandboxes.kaggle_sandbox.KaggleKernelExecutor",
        _FakeKaggleExecutor,
    )

    sandbox = KaggleSandbox(config=SandboxConfig(timeout=10.0))
    sandbox.start()

    result = sandbox.run_code("print('ok')")

    assert result.execution_ok is True
    assert result.code_error is None
    assert result.execution_count == 7
    assert "hello from kaggle" in result.stdout
    assert result.text == "42"

    sandbox.stop()


def test_kaggle_batch_mode_streaming_emits_status_and_stdout(monkeypatch):
    """run_code_streaming should emit Kaggle status updates and final output lines."""

    class _FakeStatus:
        def __init__(self, status, failure_message=None):
            self.status = status
            self.failure_message = failure_message

    class _FakeApi:
        def __init__(self):
            self._statuses = ["RUNNING", "COMPLETE"]

        def kernels_status(self, _slug):
            status = self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
            return _FakeStatus(status)

    class _FakeKaggleResult:
        slug = "demo/demo-slug"
        status = "QUEUED"
        failure_message = None
        log = None
        notebook = None
        output_dir = None
        output_files: ClassVar[list[str]] = []

        @staticmethod
        def to_kernel_reply():
            return {
                "execution_count": 1,
                "status": "ok",
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": "hello from kaggle\\n"}
                ],
            }

    class _FakeKaggleExecutor:
        def __init__(self, username=None, quiet=True):
            self.username = username
            self.quiet = quiet
            self.api = _FakeApi()

        def execute(self, code, wait=True, timeout=0.0, download_output=True, accelerator=None):
            assert "print('ok')" in code
            assert wait is False
            return _FakeKaggleResult()

        def output(self, slug, dest, force=True, quiet=None):
            _ = (slug, force, quiet)
            path = Path(dest) / "run.log"
            path.write_text("[]", encoding="utf-8")
            return [str(path)]

    monkeypatch.setattr(
        "code_sandboxes.kaggle_sandbox.KaggleKernelExecutor",
        _FakeKaggleExecutor,
    )

    sandbox = KaggleSandbox(config=SandboxConfig(timeout=10.0), poll_interval=0.0)
    sandbox.start()

    items = list(sandbox.run_code_streaming("print('ok')"))
    lines = [item.line for item in items if hasattr(item, "line")]

    assert any("submitted job" in line for line in lines)
    assert any("status: RUNNING" in line for line in lines)
    assert any("status: COMPLETE" in line for line in lines)
    assert any("hello from kaggle" in line for line in lines)

    sandbox.stop()


def test_modal_start_uses_supported_default_python_version(monkeypatch):
    """Default Modal image should pin a Modal-supported Python series."""

    class _FakeImage:
        def pip_install(self, *_args):
            return self

    class _FakeApp:
        pass

    class _FakeSandboxObj:
        object_id = "modal-object-id"

        def terminate(self):
            return None

        def detach(self):
            return None

    captured: dict = {}

    class _FakeModal:
        class App:
            @staticmethod
            def lookup(_name, create_if_missing=False):
                assert create_if_missing is True
                return _FakeApp()

        class Image:
            @staticmethod
            def debian_slim(*, python_version):
                captured["python_version"] = python_version
                return _FakeImage()

        class Secret:
            @staticmethod
            def from_dict(_values):
                return object()

        class Sandbox:
            @staticmethod
            def create(**kwargs):
                captured["create_kwargs"] = kwargs
                return _FakeSandboxObj()

    monkeypatch.setitem(sys.modules, "modal", _FakeModal)

    sandbox = ModalSandbox(config=SandboxConfig(timeout=10.0, max_lifetime=30.0))
    sandbox.start()

    assert captured["python_version"] == "3.12"
    assert sandbox.is_started is True

    sandbox.stop()


def test_modal_start_forwards_gpu_flavor(monkeypatch):
    """Configured GPU flavor should be propagated to Modal Sandbox.create."""

    class _FakeImage:
        def pip_install(self, *_args):
            return self

    class _FakeApp:
        pass

    class _FakeSandboxObj:
        object_id = "modal-object-id"

        def terminate(self):
            return None

        def detach(self):
            return None

    class _FakeGpu:
        @staticmethod
        def a100():
            return "GPU_A100"

    _FakeGpu.A100 = staticmethod(_FakeGpu.a100)

    captured: dict = {}

    class _FakeModal:
        class App:
            @staticmethod
            def lookup(_name, create_if_missing=False):
                assert create_if_missing is True
                return _FakeApp()

        class Image:
            @staticmethod
            def debian_slim(*, python_version):
                captured["python_version"] = python_version
                return _FakeImage()

        class Secret:
            @staticmethod
            def from_dict(_values):
                return object()

        class Sandbox:
            @staticmethod
            def create(**kwargs):
                captured["create_kwargs"] = kwargs
                return _FakeSandboxObj()

    _FakeModal.gpu = _FakeGpu

    monkeypatch.setitem(sys.modules, "modal", _FakeModal)

    sandbox = ModalSandbox(config=SandboxConfig(timeout=10.0, max_lifetime=30.0, gpu="A100"))
    sandbox.start()

    assert captured["create_kwargs"]["gpu"] == "GPU_A100"

    sandbox.stop()
