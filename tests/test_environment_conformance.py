# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The conformance suite against a scripted sandbox: every check passing, and each one failing.

The sandbox here answers each probe the way a real one would, so what is
checked is the suite's judgment and its messages. The probes themselves are
compiled, so a typo in the code sent to a sandbox fails here rather than on a
provider.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from code_sandboxes.contents import _MARKER
from code_sandboxes.environments.conformance import (
    CORE_CHECKS,
    EXTENDED_CHECKS,
    run_conformance,
    run_core_tier,
    run_extended_tier,
    secret_fingerprints,
)
from code_sandboxes.models import CodeError, Context, ExecutionResult, Logs, OutputMessage, Result

GOOD: dict[int, Any] = {
    1: {
        "returncode": 0,
        "stdout": json.dumps({"contract": "sandbox-contract/v1", "failed": None}),
        "stderr": "",
    },
    2: {
        "uid": 1000,
        "gid": 100,
        "user": "datalayer",
        "home": "/home/datalayer",
        "cwd": "/home/datalayer/content",
    },
    3: {"version": "3.12", "full": "3.12.6", "pip": "/opt/conda/bin/pip", "uv": None},
    5: {
        "geopandas": {"version": "1.1.1", "imported": "geopandas", "error": None},
        "scikit-learn": {"version": "1.6.1", "imported": "sklearn", "error": None},
    },
    6: {"write": True, "read": True, "delete": True, "reservedWritable": False, "error": None},
    8: {"exited": True, "seconds": 0.02, "pid1": "tini", "orphanZombies": 0},
    9: {"env": [], "files": [], "scannedBytes": 4096, "truncated": False},
    10: {"pypi.org:443": True, "169.254.169.254:80": False},
    11: {"returncode": 0, "gpus": ["NVIDIA A100-SXM4-80GB"], "cuda": "12.4"},
    12: {"mibPerSecond": 250.0},
}

PACKAGES = {"geopandas": "1.1.1", "scikit_learn": "1.6.1"}


class ScriptedSandbox:
    """Answers each tagged probe from a script; runs `1+1` and the restart marker itself."""

    def __init__(self, answers: dict[int, Any] | None = None, *, kernel: str = "2") -> None:
        self.answers = {**GOOD, **(answers or {})}
        self.kernel = kernel
        self.namespace: set[str] = set()
        self.codes: list[str] = []
        self.contexts: list[Context] = []

    def run_code(
        self, code: str, context: Context | None = None, timeout: float | None = None, **_: Any
    ) -> ExecutionResult:
        self.codes.append(code)
        tag = re.match(r"# dl-conformance: (\d+)\n", code)
        if tag:
            answer = self.answers[int(tag.group(1))]
            if isinstance(answer, BaseException):
                return ExecutionResult(
                    code_error=CodeError(name=type(answer).__name__, value=str(answer))
                )
            return ExecutionResult(
                logs=Logs(stdout=[OutputMessage(line=_MARKER + json.dumps(answer))])
            )
        if code == "1+1":
            return ExecutionResult(results=[Result(data={"text/plain": self.kernel})])
        name, assigned, _ = code.partition(" = ")
        if assigned:
            self.namespace.add(name)
            return ExecutionResult()
        asked = re.match(r"'(\w+)' in globals\(\)", code)
        if asked:
            return ExecutionResult(
                results=[Result(data={"text/plain": str(asked.group(1) in self.namespace)})]
            )
        raise AssertionError(f"unexpected code: {code!r}")

    def restart(self) -> None:
        self.namespace.clear()

    def create_context(self, name: str) -> Context:
        context = Context(id=name)
        self.contexts.append(context)
        return context


def core(sandbox: ScriptedSandbox, **options: Any):
    options.setdefault("restart", sandbox.restart)
    return run_core_tier(sandbox, python_version="3.12", expected_packages=PACKAGES, **options)


def by_id(result, check: int):
    return next(item for item in result.checks if item.id == f"conformance:{check}")


def test_a_conforming_sandbox_passes_the_core_tier() -> None:
    sandbox = ScriptedSandbox()
    result = core(sandbox, secret_values=["hunter2-token"])
    assert [item.id for item in result.checks] == [f"conformance:{check}" for check in CORE_CHECKS]
    assert result.passed, [(item.id, item.detail) for item in result.failures]
    assert all(item.gating for item in result.checks)


def test_every_probe_sent_to_a_sandbox_is_valid_python() -> None:
    sandbox = ScriptedSandbox()
    core(sandbox, secret_values=["hunter2-token"])
    run_extended_tier(
        sandbox,
        accelerator_requested=True,
        egress_allowed=["pypi.org:443"],
        egress_blocked=["169.254.169.254:80"],
    )
    probes = [code for code in sandbox.codes if code.startswith("# dl-conformance:")]
    assert len(probes) == 10
    for code in probes:
        compile(code, "<probe>", "exec")


def test_no_build_secret_is_sent_to_the_sandbox() -> None:
    sandbox = ScriptedSandbox()
    core(sandbox, secret_values=["hunter2-token"])
    assert not any("hunter2" in code for code in sandbox.codes)


@pytest.mark.parametrize(
    ("check", "answer", "detail"),
    [
        (1, {"returncode": None, "stdout": "", "stderr": "No such file"}, "not runnable"),
        (
            1,
            {
                "returncode": 1,
                "stdout": json.dumps({"contract": "sandbox-contract/v1", "failed": "user"}),
                "stderr": "",
            },
            "fails on `user`",
        ),
        (
            1,
            {
                "returncode": 0,
                "stdout": json.dumps({"contract": "sandbox-contract/v0", "failed": None}),
                "stderr": "",
            },
            "checks 'sandbox-contract/v0'",
        ),
        (1, {"returncode": 0, "stdout": "not json", "stderr": ""}, "no JSON report"),
        (
            2,
            {**GOOD[2], "user": "jovyan", "home": "/home/jovyan"},
            "user is 'jovyan', not 'datalayer'",
        ),
        (3, {**GOOD[3], "version": "3.11", "full": "3.11.9"}, "Python is 3.11.9, not 3.12"),
        (3, {**GOOD[3], "pip": None}, "neither pip nor uv"),
        (
            5,
            {**GOOD[5], "geopandas": {"version": "1.1.2", "imported": "geopandas", "error": None}},
            "geopandas is 1.1.2, the lock says 1.1.1",
        ),
        (
            5,
            {
                **GOOD[5],
                "scikit-learn": {"version": None, "imported": None, "error": "not installed"},
            },
            "scikit-learn: not installed",
        ),
        (6, {**GOOD[6], "reservedWritable": True}, "/opt/datalayer is writable"),
        (
            6,
            {**GOOD[6], "write": False, "error": "Permission denied"},
            "is not writable: Permission denied",
        ),
        (8, {**GOOD[8], "exited": False}, "ignored SIGTERM"),
        (8, {**GOOD[8], "orphanZombies": 3}, "leaves orphans unreaped"),
        (8, {**GOOD[8], "orphanZombies": None}, "cannot be observed"),
        (
            9,
            {**GOOD[9], "env": ["PIP_EXTRA_INDEX_TOKEN"]},
            "in the environment: PIP_EXTRA_INDEX_TOKEN",
        ),
        (9, {**GOOD[9], "files": ["/home/datalayer/.netrc"]}, "in a file: /home/datalayer/.netrc"),
        (2, RuntimeError("kernel died"), "kernel died"),
    ],
)
def test_each_core_check_fails_for_its_reason(check: int, answer: Any, detail: str) -> None:
    result = core(ScriptedSandbox({check: answer}), secret_values=["hunter2-token"])
    failed = by_id(result, check)
    assert not failed.passed and not result.passed
    assert detail in (failed.detail or ""), failed.detail


def test_the_kernel_must_answer_two() -> None:
    result = core(ScriptedSandbox(kernel="3"))
    assert "`1+1` gave '3'" in by_id(result, 4).detail


def test_without_a_restart_the_restart_check_fails_by_name() -> None:
    result = core(ScriptedSandbox(), restart=None)
    assert by_id(result, 7).detail == "no kernel restart was given for this sandbox"


def test_state_that_survives_a_restart_fails() -> None:
    result = core(ScriptedSandbox(), restart=lambda: None)
    assert "survived the restart" in by_id(result, 7).detail


def test_build_secrets_in_the_image_metadata_fail_without_asking_the_sandbox() -> None:
    result = core(
        ScriptedSandbox(),
        secret_values=["hunter2-token"],
        image_metadata='{"Env": ["TOKEN=hunter2-token"]}',
    )
    assert "image metadata" in by_id(result, 9).detail


def test_a_build_without_secrets_has_nothing_to_look_for() -> None:
    sandbox = ScriptedSandbox()
    result = core(sandbox)
    assert by_id(result, 9).passed
    assert not any(code.startswith("# dl-conformance: 9") for code in sandbox.codes)


def test_the_fingerprint_is_the_rolling_hash_the_sandbox_computes() -> None:
    ((length, digest),) = secret_fingerprints(["ab"])
    assert length == 2
    assert digest == (97 * 257 + 98) % ((1 << 61) - 1)
    assert secret_fingerprints([""]) == []


def test_the_extended_tier_records_without_gating() -> None:
    sandbox = ScriptedSandbox({12: {"mibPerSecond": 3.0}})
    result = run_extended_tier(sandbox, cold_start_seconds=12.5, cold_start_budget=10.0)
    assert [item.id for item in result.checks] == [
        f"conformance:{check}" for check in EXTENDED_CHECKS
    ]
    assert not any(item.gating for item in result.checks)
    assert result.passed
    assert by_id(result, 10).detail == "no egress policy to check"
    assert by_id(result, 11).detail == "no accelerator was requested"
    assert "below 20.0" in by_id(result, 12).detail
    assert "over the 10.0 s budget" in by_id(result, 13).detail
    assert by_id(result, 14).passed and len(sandbox.contexts) == 4


def test_egress_and_gpu_are_judged_against_what_was_asked() -> None:
    sandbox = ScriptedSandbox(
        {
            10: {"pypi.org:443": False, "169.254.169.254:80": True},
            11: {"returncode": 0, "gpus": ["A100"], "cuda": "12.2"},
        }
    )
    result = run_extended_tier(
        sandbox,
        accelerator_requested=True,
        cuda_version="12.4",
        egress_allowed=["pypi.org:443"],
        egress_blocked=["169.254.169.254:80"],
    )
    assert "pypi.org:443 is unreachable" in by_id(result, 10).detail
    assert "169.254.169.254:80 is reachable" in by_id(result, 10).detail
    assert "CUDA is 12.2, not 12.4" in by_id(result, 11).detail


def test_the_probes_run_for_real_in_a_local_sandbox(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The eval sandbox runs the probes in this process: they are not only valid, they work.

    The contract is this host's, so the checks that can pass here must; the
    doctor runs and fails, because a laptop is not a sandbox; and a planted
    secret is found in a file and in the environment without being sent.
    """
    import getpass
    import os
    import sys
    import uuid

    import pydantic

    from code_sandboxes.environments.contract import SANDBOX_CONTRACT_V1
    from code_sandboxes.environments.doctor.build import build_zipapp
    from code_sandboxes.eval_sandbox import EvalSandbox

    if os.geteuid() == 0:
        pytest.skip("a read-only directory is writable by root")
    workdir = tmp_path / "content"
    workdir.mkdir()
    reserved = tmp_path / "reserved"
    reserved.mkdir()
    secret = "dl-secret-" + uuid.uuid4().hex
    (workdir / "leak.txt").write_text(f"token={secret}\n")
    monkeypatch.setenv("LEAKED_TOKEN", secret)
    monkeypatch.chdir(workdir)
    doctor = build_zipapp(tmp_path / "datalayer-sandbox", interpreter=sys.executable)
    contract = SANDBOX_CONTRACT_V1.model_copy(
        update={
            "user": getpass.getuser(),
            "uid": os.getuid(),
            "gid": os.getgid(),
            "home": os.environ.get("HOME"),
            "workdir": str(workdir),
            "reserved_path": str(reserved),
            "doctor_path": str(doctor),
        }
    )
    reserved.chmod(0o555)
    sandbox = EvalSandbox()
    sandbox.start()
    try:
        result = run_core_tier(
            sandbox,
            contract=contract,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
            expected_packages={"pydantic": pydantic.VERSION},
            secret_values=[secret],
            secret_roots=[str(tmp_path)],
            restart=lambda: (sandbox.stop(), sandbox.start()),
        )
    finally:
        sandbox.stop()
        reserved.chmod(0o755)
    outcome = {item.name: (item.passed, item.detail) for item in result.checks}
    assert "fails on" in (outcome["doctor"][1] or ""), outcome["doctor"]
    for name in ("identity", "python", "kernel", "imports", "filesystem", "shutdown"):
        assert outcome[name][0], (name, outcome[name][1])
    assert not outcome["secrets"][0]
    assert "LEAKED_TOKEN" in outcome["secrets"][1] and "leak.txt" in outcome["secrets"][1]


def test_both_tiers_together_are_decided_by_the_core_tier() -> None:
    sandbox = ScriptedSandbox({12: {"mibPerSecond": 1.0}})
    result = run_conformance(
        sandbox, python_version="3.12", expected_packages=PACKAGES, restart=sandbox.restart
    )
    assert len(result.checks) == len(CORE_CHECKS) + len(EXTENDED_CHECKS)
    assert result.passed


def test_extended_naming_contract_or_timeout_does_not_crash(monkeypatch) -> None:
    """`run_extended_tier(contract=..., timeout=..., **dict(extended))` used
    to raise `got multiple values for keyword argument` the moment `extended`
    named either one itself (found on PR #27's Copilot review). `contract`
    and `timeout` are `run_conformance`'s own, shared across both tiers, so
    they win over a same-named entry in `extended`; a genuinely extended-only
    option still passes through."""
    from code_sandboxes.environments import conformance as conformance_module
    from code_sandboxes.environments.contract import SANDBOX_CONTRACT_V1

    seen: dict[str, object] = {}
    real = conformance_module.run_extended_tier

    def spying(sandbox, **kwargs):
        seen.update(kwargs)
        return real(sandbox, **kwargs)

    monkeypatch.setattr(conformance_module, "run_extended_tier", spying)
    sandbox = ScriptedSandbox({12: {"mibPerSecond": 1.0}})
    run_conformance(
        sandbox,
        python_version="3.12",
        expected_packages=PACKAGES,
        restart=sandbox.restart,
        timeout=120.0,
        extended={"timeout": 5.0, "contract": SANDBOX_CONTRACT_V1, "concurrent_kernels": 2},
    )
    assert seen["timeout"] == 120.0
    assert seen["concurrent_kernels"] == 2
