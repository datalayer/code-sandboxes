# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""datalayer-sandbox: check this sandbox against ``sandbox-contract/v1``.

    datalayer-sandbox doctor           # one line per row, exit 1 on the first failure
    datalayer-sandbox doctor --json    # the whole report as JSON on stdout

Standard library only, and no syntax newer than Python 3.8: this runs inside
images that carry none of ``code_sandboxes``' dependencies, under whatever
``python3`` the image has. A missing ``python3`` is itself a contract
failure, and the one this checker cannot report.

Every row is required. The exit code is 0 when all pass and 1 otherwise, and
the first failing row is named on stderr, so a build log says what to fix.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import ssl
import subprocess
import sys
import time
import uuid

DOCTOR = "datalayer-sandbox"
DOCTOR_VERSION = "1"

#: The contract this checker checks. Kept equal to
#: ``code_sandboxes.environments.contract.SANDBOX_CONTRACT_V1`` by a test,
#: because this module may not import it.
CONTRACT = {
    "version": "sandbox-contract/v1",
    "architecture": "linux/amd64",
    "os_family": "debian",
    "user": "datalayer",
    "uid": 1000,
    "gid": 100,
    "home": "/home/datalayer",
    "workdir": "/home/datalayer/content",
    "reserved_path": "/opt/datalayer",
    "python_executables": ["python3", "pip"],
    "kernel_packages": ["ipykernel", "jupyter_client"],
    "locale": "C.UTF-8",
    "timezone": "UTC",
}

ROW_IDS = (
    "architecture",
    "os",
    "user",
    "uid",
    "gid",
    "home",
    "workdir",
    "python",
    "pip",
    "kernel",
    "reserved_path",
    "locale",
    "timezone",
    "ca_certificates",
    "init",
)

_AMD64 = ("x86_64", "amd64")
_UTC = ("UTC", "Etc/UTC", "UTC0", "Etc/UCT", "UCT", "Universal", "Etc/Universal", "Zulu")


class Host:
    """What the checks read, in one place, so a test can hand in another."""

    def machine(self):
        return platform.machine()

    def system(self):
        return platform.system().lower()

    def os_release(self):
        values = {}
        for path in ("/etc/os-release", "/usr/lib/os-release"):
            try:
                with open(path) as handle:
                    for line in handle:
                        key, _, value = line.strip().partition("=")
                        if key:
                            values[key] = value.strip().strip('"')
                return values
            except OSError:
                continue
        return values

    def uid(self):
        return os.getuid()

    def gid(self):
        return os.getgid()

    def user(self):
        try:
            import pwd

            return pwd.getpwuid(os.getuid()).pw_name
        except (ImportError, KeyError):
            return None

    def environ(self):
        return os.environ

    def cwd(self):
        return os.getcwd()

    def which(self, name):
        return shutil.which(name)

    def python_version(self):
        return platform.python_version()

    def package_version(self, name):
        try:
            from importlib import metadata
        except ImportError:  # pragma: no cover - Python 3.7
            return None
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    def is_dir(self, path):
        return os.path.isdir(path)

    def can_write(self, path):
        """Whether a file can really be created there: permission bits lie on read-only mounts."""
        if not os.path.isdir(path):
            return False
        target = os.path.join(path, ".datalayer-doctor-" + uuid.uuid4().hex)
        try:
            with open(target, "w") as handle:
                handle.write("ok")
            os.remove(target)
            return True
        except OSError:
            return False

    def timezone(self):
        value = os.environ.get("TZ")
        if value:
            return value.lstrip(":")
        try:
            with open("/etc/timezone") as handle:
                found = handle.read().strip()
                if found:
                    return found
        except OSError:
            pass
        try:
            target = os.readlink("/etc/localtime")
            marker = "zoneinfo/"
            if marker in target:
                return target.split(marker, 1)[1]
        except OSError:
            pass
        return time.tzname[0] if time.tzname else None

    def ca_bundle(self):
        paths = ssl.get_default_verify_paths()
        for candidate in (
            paths.cafile,
            paths.openssl_cafile,
            "/etc/ssl/certs/ca-certificates.crt",
            "/etc/pki/tls/certs/ca-bundle.crt",
        ):
            if candidate and os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                return candidate
        return None

    def pid1(self):
        try:
            with open("/proc/1/comm") as handle:
                return handle.read().strip()
        except OSError:
            return None

    def orphan_zombies(self):
        """Orphans PID 1 left unreaped a moment after they exited, or None without /proc."""
        if not os.path.isdir("/proc/1"):
            return None
        try:
            # A fixed argv, run in the sandbox's own /bin/sh: not untrusted input.
            command = ["sh", "-c", "sleep 0.1 &"]
            subprocess.Popen(command).wait(timeout=5)  # noqa: S603
        except (OSError, subprocess.TimeoutExpired):
            return None
        time.sleep(0.6)
        zombies = 0
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat") as handle:
                    stat = handle.read()
            except OSError:
                continue
            # "pid (comm) state ppid ...": comm may hold spaces, so split after it.
            fields = stat.rsplit(")", 1)[-1].split()
            if len(fields) >= 2 and fields[0] == "Z" and fields[1] == "1":
                zombies += 1
        return zombies


def _row(row_id, ok, expected, actual, message=None):
    return {
        "id": row_id,
        "required": True,
        "ok": bool(ok),
        "expected": expected,
        "actual": actual,
        "message": None if ok else message,
    }


def _normal_locale(value):
    return (value or "").lower().replace("-", "").replace("_", "")


def check(host=None, contract=None):
    """The report: every row, and what the sandbox is."""
    host = host or Host()
    contract = contract or CONTRACT
    environ = host.environ()
    rows = []

    machine = host.machine()
    rows.append(
        _row(
            "architecture",
            host.system() == "linux" and machine in _AMD64,
            contract["architecture"],
            f"{host.system()}/{machine}",
            "the sandbox is not linux/amd64",
        )
    )

    release = host.os_release()
    family = [release.get("ID", ""), *release.get("ID_LIKE", "").split()]
    rows.append(
        _row(
            "os",
            "debian" in family or "ubuntu" in family,
            "Debian-derived",
            release.get("PRETTY_NAME") or release.get("ID"),
            "the base is not Debian or Ubuntu",
        )
    )

    user, uid, gid = host.user(), host.uid(), host.gid()
    rows.append(
        _row(
            "user",
            user == contract["user"],
            contract["user"],
            user,
            "runs as {}, not {}".format(user, contract["user"]),
        )
    )
    rows.append(
        _row(
            "uid",
            uid == contract["uid"],
            contract["uid"],
            uid,
            "runs as uid {}, not {}".format(uid, contract["uid"]),
        )
    )
    rows.append(
        _row(
            "gid",
            gid == contract["gid"],
            contract["gid"],
            gid,
            "runs as gid {}, not {}".format(gid, contract["gid"]),
        )
    )

    home = environ.get("HOME")
    rows.append(
        _row(
            "home",
            home == contract["home"] and host.is_dir(contract["home"]),
            contract["home"],
            home,
            "HOME is {}, not {}".format(home, contract["home"]),
        )
    )

    workdir = contract["workdir"]
    workdir_writable = host.can_write(workdir)
    rows.append(
        _row(
            "workdir",
            workdir_writable,
            workdir,
            host.cwd(),
            f"{workdir} is missing or not writable",
        )
    )

    python3 = host.which("python3")
    rows.append(
        _row("python", python3 is not None, "python3 on PATH", python3, "python3 is not on PATH")
    )
    pip = host.which("pip")
    rows.append(_row("pip", pip is not None, "pip on PATH", pip, "pip is not on PATH"))

    kernel = {name: host.package_version(name) for name in contract["kernel_packages"]}
    missing = [name for name, version in kernel.items() if version is None]
    rows.append(
        _row(
            "kernel",
            not missing,
            contract["kernel_packages"],
            kernel,
            "missing " + ", ".join(missing),
        )
    )

    reserved = contract["reserved_path"]
    reserved_present = host.is_dir(reserved)
    reserved_writable = host.can_write(reserved) if reserved_present else None
    rows.append(
        _row(
            "reserved_path",
            reserved_present and not reserved_writable,
            f"{reserved} present and read-only",
            "missing"
            if not reserved_present
            else ("writable" if reserved_writable else "read-only"),
            f"{reserved} is missing" if not reserved_present else f"{reserved} is writable",
        )
    )

    current_locale = environ.get("LC_ALL") or environ.get("LANG")
    rows.append(
        _row(
            "locale",
            _normal_locale(current_locale) == _normal_locale(contract["locale"]),
            contract["locale"],
            current_locale,
            "the locale is {}, not {}".format(current_locale, contract["locale"]),
        )
    )

    zone = host.timezone()
    rows.append(
        _row(
            "timezone",
            zone in _UTC,
            contract["timezone"],
            zone,
            f"the time zone is {zone}, not UTC",
        )
    )

    bundle = host.ca_bundle()
    rows.append(
        _row("ca_certificates", bundle is not None, "a CA bundle", bundle, "no CA certificates")
    )

    zombies = host.orphan_zombies()
    rows.append(
        _row(
            "init",
            zombies == 0,
            "PID 1 reaps orphans",
            {"pid1": host.pid1(), "orphanZombies": zombies},
            "PID 1 does not reap orphans" if zombies else "PID 1 cannot be observed",
        )
    )

    failed = next((row["id"] for row in rows if row["required"] and not row["ok"]), None)
    return {
        "doctor": DOCTOR,
        "doctorVersion": DOCTOR_VERSION,
        "contract": contract["version"],
        "ok": failed is None,
        "failed": failed,
        "rows": rows,
        "python": {"version": host.python_version(), "executable": python3, "pip": pip},
        "kernel": kernel,
        "identity": {"user": user, "uid": uid, "gid": gid, "home": home, "cwd": host.cwd()},
        "paths": {
            "writable": [workdir] if workdir_writable else [],
            "readOnly": [reserved] if reserved_present and not reserved_writable else [],
        },
        "locale": current_locale,
        "timezone": zone,
        "caBundle": bundle,
    }


def _print_rows(report, stream):
    for row in report["rows"]:
        status = "ok" if row["ok"] else "FAIL"
        stream.write(f"{status:<4} {row['id']:<16} {row['message'] or ''}\n")


def main(argv=None, host=None, stdout=None, stderr=None):
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    parser = argparse.ArgumentParser(prog=DOCTOR)
    parser.add_argument("--version", action="version", version=f"{DOCTOR} {DOCTOR_VERSION}")
    commands = parser.add_subparsers(dest="command")
    doctor = commands.add_parser("doctor", help="check this sandbox against the contract")
    doctor.add_argument("--json", action="store_true", help="write the report as JSON")
    arguments = parser.parse_args(argv)
    if arguments.command != "doctor":
        parser.print_help(stderr)
        return 2
    report = check(host)
    if arguments.json:
        stdout.write(json.dumps(report, sort_keys=True) + "\n")
    else:
        _print_rows(report, stdout)
    if report["failed"]:
        row = next(row for row in report["rows"] if row["id"] == report["failed"])
        stderr.write("{} doctor: {} failed: {}\n".format(DOCTOR, row["id"], row["message"]))
        return 1
    return 0


def run():
    """The zipapp's entry point: exit with :func:`main`'s code."""
    sys.exit(main())


if __name__ == "__main__":
    run()
