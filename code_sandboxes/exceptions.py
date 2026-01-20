# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Custom exceptions for code sandboxes."""


class SandboxError(Exception):
    """Base exception for sandbox errors."""

    pass


class SandboxTimeoutError(SandboxError):
    """Raised when code execution times out."""

    def __init__(self, timeout: float, message: str = None):
        self.timeout = timeout
        super().__init__(message or f"Code execution timed out after {timeout} seconds")


class SandboxExecutionError(SandboxError):
    """Raised when code execution fails."""

    def __init__(self, error_name: str, error_value: str, traceback: str = ""):
        self.error_name = error_name
        self.error_value = error_value
        self.traceback = traceback
        super().__init__(f"{error_name}: {error_value}")


class SandboxNotStartedError(SandboxError):
    """Raised when trying to use a sandbox that hasn't been started."""

    def __init__(self):
        super().__init__("Sandbox has not been started. Use 'with' context or call start()")


class SandboxConnectionError(SandboxError):
    """Raised when connection to remote sandbox fails."""

    def __init__(self, url: str, message: str = None):
        self.url = url
        super().__init__(message or f"Failed to connect to sandbox at {url}")


class SandboxConfigurationError(SandboxError):
    """Raised when sandbox configuration is invalid."""

    pass


class ContextNotFoundError(SandboxError):
    """Raised when a requested context does not exist."""

    def __init__(self, context_id: str):
        self.context_id = context_id
        super().__init__(f"Context '{context_id}' not found")


class VariableNotFoundError(SandboxError):
    """Raised when a requested variable does not exist."""

    def __init__(self, variable_name: str):
        self.variable_name = variable_name
        super().__init__(f"Variable '{variable_name}' not found in sandbox")


class SandboxSnapshotError(SandboxError):
    """Raised when snapshot operations fail."""

    def __init__(self, operation: str, message: str = None):
        self.operation = operation
        super().__init__(message or f"Snapshot operation '{operation}' failed")


class SandboxResourceError(SandboxError):
    """Raised when resource allocation fails (CPU, GPU, memory)."""

    def __init__(self, resource_type: str, message: str = None):
        self.resource_type = resource_type
        super().__init__(message or f"Failed to allocate resource: {resource_type}")


class SandboxAuthenticationError(SandboxError):
    """Raised when authentication with the sandbox provider fails."""

    def __init__(self, message: str = None):
        super().__init__(message or "Authentication failed")


class SandboxQuotaExceededError(SandboxError):
    """Raised when sandbox quota or limits are exceeded."""

    def __init__(self, limit_type: str, limit_value: str = None):
        self.limit_type = limit_type
        self.limit_value = limit_value
        msg = f"Quota exceeded for {limit_type}"
        if limit_value:
            msg += f": {limit_value}"
        super().__init__(msg)
