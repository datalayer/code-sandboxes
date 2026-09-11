# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The conformance suite: what an artifact passes before its version is ready.

The core tier gates promotion on every required variant, and must pass the
same way on all four; the extended tier records what may legitimately differ
between variants and gates nothing. Every check runs inside a started
:class:`~code_sandboxes.base.Sandbox` through the probe channel every
provider has — code in, one JSON line out on stdout — so the suite is the same
code on Datalayer, E2B, Daytona and Modal.

Build secrets are looked for without being sent: the sandbox receives each
secret's length and a rolling hash, and reports where a window of that
length hashes the same. A check that crashes is a check that failed, never an
exception out of the suite.
"""

from __future__ import annotations

import concurrent.futures
import re
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..contents import _answer, probe
from .builders import CheckResult, ValidationResult
from .contract import SANDBOX_CONTRACT_V1, SandboxContract

if TYPE_CHECKING:
    from ..base import Sandbox

__all__ = [
    "CORE_CHECKS",
    "EXTENDED_CHECKS",
    "run_conformance",
    "run_core_tier",
    "run_extended_tier",
    "secret_fingerprints",
]

#: Appendix B's core tier: gating on every required variant.
CORE_CHECKS: dict[int, str] = {
    1: "doctor",
    2: "identity",
    3: "python",
    4: "kernel",
    5: "imports",
    6: "filesystem",
    7: "restart",
    8: "shutdown",
    9: "secrets",
}

#: Appendix B's extended tier: recorded per variant, gating nothing.
EXTENDED_CHECKS: dict[int, str] = {
    10: "egress",
    11: "gpu",
    12: "throughput",
    13: "cold_start",
    14: "concurrent_kernels",
}

_ROLLING_PRIME = (1 << 61) - 1
_ROLLING_BASE = 257
_SECRET_ROOTS = ("/etc", "/home", "/opt", "/root", "/tmp", "/var/tmp", "/usr/local/etc")  # noqa: S108 - roots scanned inside the sandbox
_MAX_SCANNED_FILE_BYTES = 1024 * 1024
_MAX_SCANNED_BYTES = 16 * 1024 * 1024

#: Drops every `_dl_` name a probe left in the namespace the caller keeps using.
_CLEANUP = (
    "for _dl_name in [_dl_n for _dl_n in list(globals()) if _dl_n.startswith('_dl_')]:\n"
    "    globals().pop(_dl_name, None)\n"
    "globals().pop('_dl_name', None)\n"
)


def secret_fingerprints(values: Sequence[str]) -> list[list[int]]:
    """Each secret as ``[length, rolling hash]``: what the sandbox searches for.

    The hash is the polynomial rolling hash the sandbox computes over every
    window of the same length; a 61-bit modulus makes a false match a
    practical impossibility, and nothing of the secret itself leaves the host.
    """
    fingerprints: list[list[int]] = []
    for value in values:
        data = value.encode("utf-8")
        if not data:
            continue
        digest = 0
        for byte in data:
            digest = (digest * _ROLLING_BASE + byte) % _ROLLING_PRIME
        fingerprints.append([len(data), digest])
    return fingerprints


def _normal_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _code(check: int, body: str, answer: str) -> str:
    return f"# dl-conformance: {check}\n" + body + _answer(answer) + _CLEANUP


def _result(
    check: int, passed: bool, *, gating: bool, detail: str | None = None, **data: Any
) -> CheckResult:
    name = CORE_CHECKS.get(check) or EXTENDED_CHECKS[check]
    return CheckResult(
        id=f"conformance:{check}",
        name=name,
        passed=passed,
        gating=gating,
        detail=None if passed and detail is None else detail,
        data=data,
    )


def _guard(check: int, gating: bool, run: Callable[[], CheckResult]) -> CheckResult:
    try:
        return run()
    except Exception as error:
        return _result(check, False, gating=gating, detail=f"{type(error).__name__}: {error}")


# --- The core tier ----------------------------------------------------------------


def _doctor(sandbox: Sandbox, contract: SandboxContract, timeout: float | None) -> CheckResult:
    import json

    body = (
        "import subprocess as _dl_sp\n"
        "try:\n"
        f"    _dl_run = _dl_sp.run([{contract.doctor_path!r}, 'doctor', '--json'], "
        "capture_output=True, text=True, timeout=120)\n"
        "    _dl_out = {'returncode': _dl_run.returncode, 'stdout': _dl_run.stdout[-65536:], "
        "'stderr': _dl_run.stderr[-4096:]}\n"
        "except (OSError, _dl_sp.SubprocessError) as _dl_error:\n"
        "    _dl_out = {'returncode': None, 'stdout': '', 'stderr': str(_dl_error)}\n"
    )
    answer = probe(sandbox, _code(1, body, "_dl_out"), timeout=timeout)
    returncode = answer.get("returncode")
    if returncode is None:
        return _result(
            1,
            False,
            gating=True,
            detail=f"the doctor is not runnable at {contract.doctor_path}: {answer.get('stderr')}",
        )
    try:
        report = json.loads(answer.get("stdout") or "")
    except ValueError:
        return _result(
            1, False, gating=True, detail="the doctor wrote no JSON report", returncode=returncode
        )
    failed = report.get("failed")
    if report.get("contract") != contract.version:
        return _result(
            1,
            False,
            gating=True,
            detail=f"the doctor checks {report.get('contract')!r}, not {contract.version!r}",
            failed=failed,
        )
    if returncode != 0:
        return _result(
            1, False, gating=True, detail=f"the doctor fails on `{failed}`", failed=failed
        )
    return _result(1, True, gating=True, failed=None)


def _identity(sandbox: Sandbox, contract: SandboxContract, timeout: float | None) -> CheckResult:
    body = (
        "import os as _dl_os\n"
        "try:\n"
        "    import pwd as _dl_pwd\n"
        "    _dl_user = _dl_pwd.getpwuid(_dl_os.getuid()).pw_name\n"
        "except Exception:\n"
        "    _dl_user = None\n"
        "_dl_out = {'uid': _dl_os.getuid(), 'gid': _dl_os.getgid(), 'user': _dl_user, "
        "'home': _dl_os.environ.get('HOME'), 'cwd': _dl_os.getcwd()}\n"
    )
    answer = probe(sandbox, _code(2, body, "_dl_out"), timeout=timeout)
    expected = {
        "user": contract.user,
        "uid": contract.uid,
        "gid": contract.gid,
        "home": contract.home,
        "cwd": contract.workdir,
    }
    wrong = {key: answer.get(key) for key, value in expected.items() if answer.get(key) != value}
    detail = ", ".join(f"{key} is {value!r}, not {expected[key]!r}" for key, value in wrong.items())
    return _result(2, not wrong, gating=True, detail=detail or None, actual=answer)


def _python(sandbox: Sandbox, python_version: str, timeout: float | None) -> CheckResult:
    body = (
        "import shutil as _dl_shutil, sys as _dl_sys\n"
        "_dl_out = {'version': '%d.%d' % _dl_sys.version_info[:2], "
        "'full': _dl_sys.version.split()[0], 'pip': _dl_shutil.which('pip'), "
        "'uv': _dl_shutil.which('uv')}\n"
    )
    answer = probe(sandbox, _code(3, body, "_dl_out"), timeout=timeout)
    problems = []
    if answer.get("version") != python_version:
        problems.append(f"Python is {answer.get('full')}, not {python_version}")
    if not (answer.get("pip") or answer.get("uv")):
        problems.append("neither pip nor uv is on PATH")
    return _result(3, not problems, gating=True, detail="; ".join(problems) or None, actual=answer)


def _kernel(sandbox: Sandbox, timeout: float | None) -> CheckResult:
    execution = sandbox.run_code("1+1", timeout=timeout)
    text = (execution.text or "").strip()
    if execution.success and text == "2":
        return _result(4, True, gating=True)
    reason = execution.execution_error or (
        execution.code_error.value if execution.code_error else None
    )
    return _result(
        4, False, gating=True, detail=f"`1+1` gave {text!r}" + (f": {reason}" if reason else "")
    )


def _imports(sandbox: Sandbox, expected: Mapping[str, str], timeout: float | None) -> CheckResult:
    if not expected:
        return _result(5, True, gating=True, detail="the lock names no top-level package")
    wanted = {_normal_distribution(name): version for name, version in expected.items()}
    body = (
        "import importlib as _dl_il\n"
        "import importlib.metadata as _dl_md\n"
        "import re as _dl_re\n"
        f"_dl_expected = {wanted!r}\n"
        "_dl_norm = lambda _dl_v: _dl_re.sub(r'[-_.]+', '-', _dl_v).lower()\n"
        "_dl_modules = {}\n"
        "for _dl_module, _dl_dists in _dl_md.packages_distributions().items():\n"
        "    for _dl_dist in _dl_dists:\n"
        "        if not _dl_module.startswith('_'):\n"
        "            _dl_modules.setdefault(_dl_norm(_dl_dist), []).append(_dl_module)\n"
        "_dl_out = {}\n"
        "for _dl_dist in _dl_expected:\n"
        "    _dl_entry = {'version': None, 'imported': None, 'error': None}\n"
        "    try:\n"
        "        _dl_entry['version'] = _dl_md.version(_dl_dist)\n"
        "    except _dl_md.PackageNotFoundError:\n"
        "        _dl_entry['error'] = 'not installed'\n"
        "    for _dl_module in sorted(_dl_modules.get(_dl_dist, []), key=len)[:1]:\n"
        "        try:\n"
        "            _dl_il.import_module(_dl_module)\n"
        "            _dl_entry['imported'] = _dl_module\n"
        "        except Exception as _dl_error:\n"
        "            _dl_entry['error'] = type(_dl_error).__name__ + ': ' + str(_dl_error)\n"
        "    _dl_out[_dl_dist] = _dl_entry\n"
    )
    answer = probe(sandbox, _code(5, body, "_dl_out"), timeout=timeout)
    problems: list[str] = []
    for name, version in wanted.items():
        found = answer.get(name) or {}
        if found.get("error"):
            problems.append(f"{name}: {found['error']}")
        elif found.get("version") != version:
            problems.append(f"{name} is {found.get('version')}, the lock says {version}")
        elif not found.get("imported"):
            problems.append(f"{name} installs no importable module")
    return _result(
        5, not problems, gating=True, detail="; ".join(problems) or None, packages=answer
    )


def _filesystem(sandbox: Sandbox, contract: SandboxContract, timeout: float | None) -> CheckResult:
    body = (
        "import os as _dl_os, uuid as _dl_uuid\n"
        "_dl_out = {'write': False, 'read': False, 'delete': False, "
        "'reservedWritable': None, 'error': None}\n"
        f"_dl_path = _dl_os.path.join({contract.workdir!r}, "
        "'.dl-conformance-' + _dl_uuid.uuid4().hex)\n"
        "try:\n"
        "    with open(_dl_path, 'w') as _dl_handle:\n"
        "        _dl_handle.write('ok')\n"
        "    _dl_out['write'] = True\n"
        "    with open(_dl_path) as _dl_handle:\n"
        "        _dl_out['read'] = _dl_handle.read() == 'ok'\n"
        "    _dl_os.remove(_dl_path)\n"
        "    _dl_out['delete'] = not _dl_os.path.exists(_dl_path)\n"
        "except OSError as _dl_error:\n"
        "    _dl_out['error'] = str(_dl_error)\n"
        f"_dl_reserved = _dl_os.path.join({contract.reserved_path!r}, "
        "'.dl-conformance-' + _dl_uuid.uuid4().hex)\n"
        "try:\n"
        "    with open(_dl_reserved, 'w') as _dl_handle:\n"
        "        _dl_handle.write('no')\n"
        "    _dl_os.remove(_dl_reserved)\n"
        "    _dl_out['reservedWritable'] = True\n"
        "except OSError:\n"
        "    _dl_out['reservedWritable'] = False\n"
    )
    answer = probe(sandbox, _code(6, body, "_dl_out"), timeout=timeout)
    problems = []
    if not (answer.get("write") and answer.get("read") and answer.get("delete")):
        problems.append(f"{contract.workdir} is not writable: {answer.get('error')}")
    if answer.get("reservedWritable"):
        problems.append(f"{contract.reserved_path} is writable")
    return _result(6, not problems, gating=True, detail="; ".join(problems) or None, actual=answer)


def _restart(
    sandbox: Sandbox, restart: Callable[[], None] | None, timeout: float | None
) -> CheckResult:
    if restart is None:
        return _result(7, False, gating=True, detail="no kernel restart was given for this sandbox")
    marker = "_dl_conformance_restart_marker"
    sandbox.run_code(f"{marker} = 1", timeout=timeout)
    restart()
    execution = sandbox.run_code(f"{marker!r} in globals()", timeout=timeout)
    text = (execution.text or "").strip()
    if execution.success and text == "False":
        return _result(7, True, gating=True)
    return _result(7, False, gating=True, detail=f"state survived the restart ({text!r})")


def _shutdown(sandbox: Sandbox, contract: SandboxContract, timeout: float | None) -> CheckResult:
    grace = contract.graceful_shutdown_seconds
    body = (
        "import os as _dl_os, signal as _dl_signal, subprocess as _dl_sp, time as _dl_time\n"
        "_dl_out = {'exited': None, 'seconds': None, 'pid1': None, 'orphanZombies': None}\n"
        "_dl_proc = _dl_sp.Popen(['sh', '-c', 'trap \"exit 0\" TERM; sleep 30 & wait'], "
        "start_new_session=True)\n"
        "_dl_time.sleep(0.3)\n"
        "_dl_start = _dl_time.monotonic()\n"
        "_dl_os.killpg(_dl_proc.pid, _dl_signal.SIGTERM)\n"
        "try:\n"
        f"    _dl_proc.wait(timeout={grace})\n"
        "    _dl_out['exited'] = True\n"
        "    _dl_out['seconds'] = round(_dl_time.monotonic() - _dl_start, 3)\n"
        "except _dl_sp.TimeoutExpired:\n"
        "    _dl_out['exited'] = False\n"
        "    _dl_os.killpg(_dl_proc.pid, _dl_signal.SIGKILL)\n"
        "try:\n"
        "    with open('/proc/1/comm') as _dl_handle:\n"
        "        _dl_out['pid1'] = _dl_handle.read().strip()\n"
        "except OSError:\n"
        "    pass\n"
        "if _dl_os.path.isdir('/proc/1'):\n"
        "    _dl_sp.Popen(['sh', '-c', 'sleep 0.1 &']).wait(timeout=5)\n"
        "    _dl_time.sleep(0.6)\n"
        "    _dl_zombies = 0\n"
        "    for _dl_entry in _dl_os.listdir('/proc'):\n"
        "        if not _dl_entry.isdigit():\n"
        "            continue\n"
        "        try:\n"
        "            with open('/proc/' + _dl_entry + '/stat') as _dl_handle:\n"
        "                _dl_fields = _dl_handle.read().rsplit(')', 1)[-1].split()\n"
        "        except OSError:\n"
        "            continue\n"
        "        if len(_dl_fields) >= 2 and _dl_fields[0] == 'Z' and _dl_fields[1] == '1':\n"
        "            _dl_zombies += 1\n"
        "    _dl_out['orphanZombies'] = _dl_zombies\n"
    )
    answer = probe(sandbox, _code(8, body, "_dl_out"), timeout=timeout)
    problems = []
    if not answer.get("exited"):
        problems.append(f"a process group ignored SIGTERM for {grace} seconds")
    if answer.get("orphanZombies") is None:
        problems.append("PID 1 cannot be observed")
    elif answer["orphanZombies"]:
        problems.append(f"PID 1 ({answer.get('pid1')}) leaves orphans unreaped")
    return _result(8, not problems, gating=True, detail="; ".join(problems) or None, actual=answer)


def _secrets(
    sandbox: Sandbox,
    secret_values: Sequence[str],
    image_metadata: str | None,
    roots: Sequence[str],
    timeout: float | None,
) -> CheckResult:
    if not secret_values:
        return _result(9, True, gating=True, detail="the build used no secret")
    in_metadata = bool(image_metadata) and any(
        value and value in image_metadata for value in secret_values
    )
    fingerprints: dict[int, list[int]] = {}
    for length, digest in secret_fingerprints(secret_values):
        fingerprints.setdefault(length, []).append(digest)
    body = (
        "import os as _dl_os\n"
        f"_dl_targets = {{_dl_k: set(_dl_v) for _dl_k, _dl_v in {fingerprints!r}.items()}}\n"
        f"_dl_p, _dl_b = {_ROLLING_PRIME}, {_ROLLING_BASE}\n"
        "def _dl_hit(_dl_data):\n"
        "    for _dl_len, _dl_hashes in _dl_targets.items():\n"
        "        if len(_dl_data) < _dl_len:\n"
        "            continue\n"
        "        _dl_power = pow(_dl_b, _dl_len - 1, _dl_p)\n"
        "        _dl_h = 0\n"
        "        for _dl_i in range(_dl_len):\n"
        "            _dl_h = (_dl_h * _dl_b + _dl_data[_dl_i]) % _dl_p\n"
        "        if _dl_h in _dl_hashes:\n"
        "            return True\n"
        "        for _dl_i in range(_dl_len, len(_dl_data)):\n"
        "            _dl_h = ((_dl_h - _dl_data[_dl_i - _dl_len] * _dl_power) * _dl_b "
        "+ _dl_data[_dl_i]) % _dl_p\n"
        "            if _dl_h in _dl_hashes:\n"
        "                return True\n"
        "    return False\n"
        "_dl_out = {'env': [], 'files': [], 'scannedBytes': 0, 'truncated': False}\n"
        "for _dl_key, _dl_value in _dl_os.environ.items():\n"
        "    if _dl_hit(_dl_value.encode('utf-8', 'surrogateescape')):\n"
        "        _dl_out['env'].append(_dl_key)\n"
        f"for _dl_root in {list(roots)!r}:\n"
        "    for _dl_dir, _dl_subdirs, _dl_files in _dl_os.walk(_dl_root):\n"
        "        for _dl_file in _dl_files:\n"
        "            _dl_path = _dl_os.path.join(_dl_dir, _dl_file)\n"
        "            try:\n"
        "                if _dl_os.path.islink(_dl_path) or not _dl_os.path.isfile(_dl_path) "
        f"or _dl_os.path.getsize(_dl_path) > {_MAX_SCANNED_FILE_BYTES}:\n"
        "                    continue\n"
        f"                if _dl_out['scannedBytes'] > {_MAX_SCANNED_BYTES}:\n"
        "                    _dl_out['truncated'] = True\n"
        "                    break\n"
        "                with open(_dl_path, 'rb') as _dl_handle:\n"
        "                    _dl_data = _dl_handle.read()\n"
        "            except OSError:\n"
        "                continue\n"
        "            _dl_out['scannedBytes'] += len(_dl_data)\n"
        "            if _dl_hit(_dl_data):\n"
        "                _dl_out['files'].append(_dl_path)\n"
    )
    answer = probe(sandbox, _code(9, body, "_dl_out"), timeout=timeout)
    problems = []
    if answer.get("env"):
        problems.append("a build secret is in the environment: " + ", ".join(sorted(answer["env"])))
    if answer.get("files"):
        problems.append("a build secret is in a file: " + ", ".join(sorted(answer["files"])[:10]))
    if in_metadata:
        problems.append("a build secret is in the image metadata")
    return _result(
        9,
        not problems,
        gating=True,
        detail="; ".join(problems) or None,
        scannedBytes=answer.get("scannedBytes"),
        truncated=answer.get("truncated"),
    )


def run_core_tier(
    sandbox: Sandbox,
    *,
    python_version: str,
    expected_packages: Mapping[str, str],
    contract: SandboxContract = SANDBOX_CONTRACT_V1,
    secret_values: Sequence[str] = (),
    image_metadata: str | None = None,
    restart: Callable[[], None] | None = None,
    secret_roots: Sequence[str] = _SECRET_ROOTS,
    timeout: float | None = 120.0,
) -> ValidationResult:
    """Appendix B checks 1-9 against a started sandbox.

    ``expected_packages`` maps each top-level distribution of the spec to the
    version its lock pins. ``secret_values`` are the build's secrets, never
    sent to the sandbox, and looked for under ``secret_roots``; ``restart``
    restarts the sandbox's kernel.
    """
    checks = [
        _guard(1, True, lambda: _doctor(sandbox, contract, timeout)),
        _guard(2, True, lambda: _identity(sandbox, contract, timeout)),
        _guard(3, True, lambda: _python(sandbox, python_version, timeout)),
        _guard(4, True, lambda: _kernel(sandbox, timeout)),
        _guard(5, True, lambda: _imports(sandbox, expected_packages, timeout)),
        _guard(6, True, lambda: _filesystem(sandbox, contract, timeout)),
        _guard(7, True, lambda: _restart(sandbox, restart, timeout)),
        _guard(8, True, lambda: _shutdown(sandbox, contract, timeout)),
        _guard(
            9, True, lambda: _secrets(sandbox, secret_values, image_metadata, secret_roots, timeout)
        ),
    ]
    return ValidationResult(contract_version=contract.version, checks=checks)


# --- The extended tier ----------------------------------------------------------


def _egress(
    sandbox: Sandbox, allowed: Sequence[str], blocked: Sequence[str], timeout: float | None
) -> CheckResult:
    if not allowed and not blocked:
        return _result(10, True, gating=False, detail="no egress policy to check")
    targets = {"allowed": list(allowed), "blocked": list(blocked)}
    body = (
        "import socket as _dl_socket\n"
        f"_dl_targets = {targets!r}\n"
        "_dl_out = {}\n"
        "for _dl_group, _dl_hosts in _dl_targets.items():\n"
        "    for _dl_host in _dl_hosts:\n"
        "        _dl_name, _, _dl_port = _dl_host.rpartition(':')\n"
        "        try:\n"
        "            _dl_socket.create_connection((_dl_name, int(_dl_port)), timeout=3).close()\n"
        "            _dl_out[_dl_host] = True\n"
        "        except OSError:\n"
        "            _dl_out[_dl_host] = False\n"
    )
    answer = probe(sandbox, _code(10, body, "_dl_out"), timeout=timeout)
    problems = [f"{host} is unreachable" for host in allowed if not answer.get(host)]
    problems += [f"{host} is reachable" for host in blocked if answer.get(host)]
    return _result(
        10, not problems, gating=False, detail="; ".join(problems) or None, reachable=answer
    )


def _gpu(sandbox: Sandbox, requested: bool, cuda: str | None, timeout: float | None) -> CheckResult:
    if not requested:
        return _result(11, True, gating=False, detail="no accelerator was requested")
    body = (
        "import re as _dl_re, subprocess as _dl_sp\n"
        "_dl_out = {'returncode': None, 'gpus': [], 'cuda': None}\n"
        "try:\n"
        "    _dl_run = _dl_sp.run(['nvidia-smi'], capture_output=True, text=True, timeout=30)\n"
        "    _dl_out['returncode'] = _dl_run.returncode\n"
        "    _dl_match = _dl_re.search(r'CUDA Version:\\s*([0-9.]+)', _dl_run.stdout)\n"
        "    _dl_out['cuda'] = _dl_match.group(1) if _dl_match else None\n"
        "    _dl_names = _dl_sp.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], "
        "capture_output=True, text=True, timeout=30)\n"
        "    _dl_out['gpus'] = [_dl_l.strip() for _dl_l in _dl_names.stdout.splitlines() "
        "if _dl_l.strip()]\n"
        "except (OSError, _dl_sp.SubprocessError) as _dl_error:\n"
        "    _dl_out['error'] = str(_dl_error)\n"
    )
    answer = probe(sandbox, _code(11, body, "_dl_out"), timeout=timeout)
    problems = []
    if answer.get("returncode") != 0 or not answer.get("gpus"):
        problems.append("no GPU is visible")
    if cuda and not str(answer.get("cuda") or "").startswith(cuda):
        problems.append(f"CUDA is {answer.get('cuda')}, not {cuda}")
    return _result(
        11, not problems, gating=False, detail="; ".join(problems) or None, actual=answer
    )


def _throughput(
    sandbox: Sandbox, contract: SandboxContract, minimum: float, timeout: float | None
) -> CheckResult:
    body = (
        "import os as _dl_os, time as _dl_time, uuid as _dl_uuid\n"
        f"_dl_path = _dl_os.path.join({contract.workdir!r}, "
        "'.dl-throughput-' + _dl_uuid.uuid4().hex)\n"
        "_dl_chunk = bytes(1024 * 1024)\n"
        "_dl_start = _dl_time.monotonic()\n"
        "with open(_dl_path, 'wb') as _dl_handle:\n"
        "    for _dl_i in range(64):\n"
        "        _dl_handle.write(_dl_chunk)\n"
        "    _dl_handle.flush()\n"
        "    _dl_os.fsync(_dl_handle.fileno())\n"
        "_dl_elapsed = max(_dl_time.monotonic() - _dl_start, 1e-6)\n"
        "_dl_os.remove(_dl_path)\n"
        "_dl_out = {'mibPerSecond': round(64 / _dl_elapsed, 1)}\n"
    )
    answer = probe(sandbox, _code(12, body, "_dl_out"), timeout=timeout)
    speed = float(answer.get("mibPerSecond") or 0)
    return _result(
        12,
        speed >= minimum,
        gating=False,
        detail=None if speed >= minimum else f"{speed} MiB/s, below {minimum}",
        mibPerSecond=speed,
    )


def _cold_start(seconds: float | None, budget: float | None) -> CheckResult:
    if seconds is None:
        return _result(13, True, gating=False, detail="cold start was not measured")
    passed = budget is None or seconds <= budget
    return _result(
        13,
        passed,
        gating=False,
        detail=None if passed else f"{seconds} s, over the {budget} s budget",
        seconds=seconds,
    )


def _concurrent(sandbox: Sandbox, kernels: int, timeout: float | None) -> CheckResult:
    contexts = [sandbox.create_context(f"dl-conformance-{index}") for index in range(kernels)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=kernels) as pool:
        results = list(
            pool.map(
                lambda context: sandbox.run_code("1+1", context=context, timeout=timeout), contexts
            )
        )
    answered = sum(1 for execution in results if (execution.text or "").strip() == "2")
    return _result(
        14,
        answered == kernels,
        gating=False,
        detail=None if answered == kernels else f"{answered} of {kernels} contexts answered",
        answered=answered,
    )


def run_extended_tier(
    sandbox: Sandbox,
    *,
    contract: SandboxContract = SANDBOX_CONTRACT_V1,
    accelerator_requested: bool = False,
    cuda_version: str | None = None,
    egress_allowed: Sequence[str] = (),
    egress_blocked: Sequence[str] = (),
    cold_start_seconds: float | None = None,
    cold_start_budget: float | None = None,
    minimum_mib_per_second: float = 20.0,
    concurrent_kernels: int = 4,
    timeout: float | None = 120.0,
) -> ValidationResult:
    """Appendix B checks 10-14: recorded per variant, gating nothing."""
    checks = [
        _guard(10, False, lambda: _egress(sandbox, egress_allowed, egress_blocked, timeout)),
        _guard(11, False, lambda: _gpu(sandbox, accelerator_requested, cuda_version, timeout)),
        _guard(12, False, lambda: _throughput(sandbox, contract, minimum_mib_per_second, timeout)),
        _guard(13, False, lambda: _cold_start(cold_start_seconds, cold_start_budget)),
        _guard(14, False, lambda: _concurrent(sandbox, concurrent_kernels, timeout)),
    ]
    return ValidationResult(contract_version=contract.version, checks=checks)


def run_conformance(
    sandbox: Sandbox,
    *,
    python_version: str,
    expected_packages: Mapping[str, str],
    contract: SandboxContract = SANDBOX_CONTRACT_V1,
    secret_values: Sequence[str] = (),
    image_metadata: str | None = None,
    restart: Callable[[], None] | None = None,
    secret_roots: Sequence[str] = _SECRET_ROOTS,
    extended: Mapping[str, Any] | None = None,
    timeout: float | None = 120.0,
) -> ValidationResult:
    """Both tiers: the core tier decides, the extended tier is recorded beside it."""
    core = run_core_tier(
        sandbox,
        python_version=python_version,
        expected_packages=expected_packages,
        contract=contract,
        secret_values=secret_values,
        image_metadata=image_metadata,
        restart=restart,
        secret_roots=secret_roots,
        timeout=timeout,
    )
    recorded = run_extended_tier(
        sandbox, contract=contract, timeout=timeout, **dict(extended or {})
    )
    return ValidationResult(contract_version=contract.version, checks=core.checks + recorded.checks)
