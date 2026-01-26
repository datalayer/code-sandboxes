# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Model tests for code-sandboxes package."""

from code_sandboxes.models import (
    CodeError,
    Context,
    ExecutionResult,
    GPUType,
    Logs,
    MIMEType,
    OutputMessage,
    ResourceConfig,
    Result,
    SandboxConfig,
    SandboxInfo,
    SandboxStatus,
    SandboxVariant as SandboxVariantEnum,
)


class TestModels:
    """Tests for data models."""

    def test_mime_type_enum(self):
        """Test MIMEType enum values."""
        assert MIMEType.TEXT_PLAIN.value == "text/plain"
        assert MIMEType.APPLICATION_JSON.value == "application/json"
        assert MIMEType.IMAGE_PNG.value == "image/png"

    def test_sandbox_status_enum(self):
        """Test SandboxStatus enum values."""
        assert SandboxStatus.PENDING.value == "pending"
        assert SandboxStatus.RUNNING.value == "running"
        assert SandboxStatus.STOPPED.value == "stopped"
        assert SandboxStatus.ERROR.value == "error"

    def test_sandbox_variant_enum(self):
        """Test SandboxVariant enum values."""
        assert SandboxVariantEnum.LOCAL_EVAL.value == "local-eval"
        assert SandboxVariantEnum.LOCAL_DOCKER.value == "local-docker"
        assert SandboxVariantEnum.LOCAL_JUPYTER.value == "local-jupyter"
        assert SandboxVariantEnum.DATALAYER_RUNTIME.value == "datalayer-runtime"

    def test_gpu_type_enum(self):
        """Test GPUType enum values."""
        assert GPUType.T4.value == "T4"
        assert GPUType.A100.value == "A100"
        assert GPUType.H100.value == "H100"

    def test_resource_config(self):
        """Test ResourceConfig dataclass."""
        config = ResourceConfig(
            cpu=2.0,
            memory=4096,
            gpu="T4",
            gpu_count=1,
        )

        assert config.cpu == 2.0
        assert config.memory == 4096
        assert config.gpu == "T4"
        assert "cpu=2.0" in repr(config)

    def test_resource_config_default(self):
        """Test ResourceConfig with defaults."""
        config = ResourceConfig()

        assert config.cpu is None
        assert config.memory is None
        assert config.gpu is None
        assert config.gpu_count == 1

    def test_output_message(self):
        """Test OutputMessage dataclass."""
        msg = OutputMessage(
            line="Hello, World!",
            timestamp=1234567890.0,
            error=False,
        )

        assert msg.line == "Hello, World!"
        assert msg.timestamp == 1234567890.0
        assert msg.error is False

    def test_output_message_defaults(self):
        """Test OutputMessage with defaults."""
        msg = OutputMessage(line="test")

        assert msg.line == "test"
        assert msg.timestamp == 0.0
        assert msg.error is False

    def test_logs_text_helpers(self):
        """Test Logs stdout/stderr text helpers."""
        logs = Logs(
            stdout=[OutputMessage(line="line1"), OutputMessage(line="line2")],
            stderr=[OutputMessage(line="err1", error=True)],
        )

        assert logs.stdout_text == "line1\nline2"
        assert logs.stderr_text == "err1"

    def test_result(self):
        """Test Result dataclass."""
        result = Result(
            data={"text/plain": "42"},
            is_main_result=True,
        )

        assert result.data["text/plain"] == "42"
        assert result.is_main_result is True

    def test_code_error(self):
        """Test CodeError dataclass."""
        error = CodeError(
            name="ValueError",
            value="Invalid input",
            traceback="Traceback...",
        )

        assert error.name == "ValueError"
        assert error.value == "Invalid input"
        assert error.traceback == "Traceback..."

    def test_execution_success_status(self):
        """Test Execution with successful execution."""
        execution = ExecutionResult(
            results=[Result(data={"text/plain": "42"})],
            execution_ok=True,
            code_error=None,
            started_at=1000.0,
            completed_at=1001.5,
        )

        assert execution.execution_ok is True
        assert execution.execution_error is None
        assert execution.code_error is None
        assert execution.success is True
        assert execution.duration == 1.5

    def test_execution_code_error(self):
        """Test Execution with code error (Python exception)."""
        execution = ExecutionResult(
            execution_ok=True,
            code_error=CodeError(
                name="ValueError",
                value="Invalid value",
                traceback="Traceback...",
            ),
        )

        assert execution.execution_ok is True
        assert execution.code_error is not None
        assert execution.code_error.name == "ValueError"
        assert execution.success is False

    def test_execution_infrastructure_failure(self):
        """Test Execution with infrastructure failure."""
        execution = ExecutionResult(
            execution_ok=False,
            execution_error="Connection timeout",
        )

        assert execution.execution_ok is False
        assert execution.execution_error == "Connection timeout"
        assert execution.code_error is None
        assert execution.success is False

    def test_execution_interrupted(self):
        """Test Execution that was interrupted."""
        execution = ExecutionResult(
            execution_ok=True,
            interrupted=True,
        )

        assert execution.execution_ok is True
        assert execution.interrupted is True
        assert execution.success is False

    def test_execution_exit_code_failure(self):
        """Test Execution with non-zero exit code."""
        execution = ExecutionResult(
            execution_ok=True,
            exit_code=2,
        )

        assert execution.execution_ok is True
        assert execution.exit_code == 2
        assert execution.code_error is None
        assert execution.success is False

    def test_execution_exit_code_success(self):
        """Test Execution with zero exit code."""
        execution = ExecutionResult(
            execution_ok=True,
            exit_code=0,
        )

        assert execution.execution_ok is True
        assert execution.exit_code == 0
        assert execution.success is True

    def test_sandbox_config(self):
        """Test SandboxConfig dataclass."""
        config = SandboxConfig(
            timeout=60.0,
            environment="python-gpu-env",
        )

        assert config.timeout == 60.0
        assert config.environment == "python-gpu-env"

    def test_context(self):
        """Test Context dataclass."""
        ctx = Context(id="ctx-123", language="python")

        assert ctx.id == "ctx-123"
        assert ctx.language == "python"

    def test_sandbox_info(self):
        """Test SandboxInfo model usage."""
        info = SandboxInfo(
            id="sandbox-123",
            variant="local-eval",
            status=SandboxStatus.RUNNING,
            created_at=1234567890.0,
            name="test-sandbox",
            metadata={"owner": "local"},
            config=SandboxConfig(timeout=45.0),
        )

        assert info.id == "sandbox-123"
        assert info.variant == "local-eval"
        assert info.status == SandboxStatus.RUNNING
        assert info.created_at == 1234567890.0
        assert info.name == "test-sandbox"
        assert info.metadata == {"owner": "local"}
        assert info.config.timeout == 45.0
