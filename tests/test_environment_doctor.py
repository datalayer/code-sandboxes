# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The doctor: its report pinned by a schema, its rows judged on fake hosts, its zipapp run."""

from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest

from code_sandboxes.environments.contract import SANDBOX_CONTRACT_V1
from code_sandboxes.environments.doctor import datalayer_sandbox as doctor
from code_sandboxes.environments.doctor.build import build_zipapp

SCHEMA = Path(doctor.__file__).with_name("report.schema.json")


class CompliantHost(doctor.Host):
    """A sandbox that satisfies the contract, as a report would see it."""

    environment: ClassVar[dict[str, str]] = {"HOME": "/home/datalayer", "LANG": "C.UTF-8"}
    directories: ClassVar[set[str]] = {
        "/home/datalayer",
        "/home/datalayer/content",
        "/opt/datalayer",
    }
    writable: ClassVar[set[str]] = {"/home/datalayer/content"}

    def machine(self):
        return "x86_64"

    def system(self):
        return "linux"

    def os_release(self):
        return {"ID": "ubuntu", "ID_LIKE": "debian", "PRETTY_NAME": "Ubuntu 24.04"}

    def uid(self):
        return 1000

    def gid(self):
        return 100

    def user(self):
        return "datalayer"

    def environ(self):
        return self.environment

    def cwd(self):
        return "/home/datalayer/content"

    def which(self, name):
        return "/opt/conda/bin/" + name

    def python_version(self):
        return "3.12.6"

    def package_version(self, name):
        return {"ipykernel": "6.29.5", "jupyter_client": "8.6.3"}.get(name)

    def is_dir(self, path):
        return path in self.directories

    def can_write(self, path):
        return path in self.writable

    def timezone(self):
        return "Etc/UTC"

    def ca_bundle(self):
        return "/etc/ssl/certs/ca-certificates.crt"

    def pid1(self):
        return "tini"

    def orphan_zombies(self):
        return 0


class JupyterPython011(CompliantHost):
    """`jupyter-python:0.1.1` as the doctor found it on 2026-09-11."""

    environment: ClassVar[dict[str, str]] = {"HOME": "/home/jovyan", "LANG": "C.UTF-8"}
    directories: ClassVar[set[str]] = {"/home/jovyan"}
    writable: ClassVar[set[str]] = {"/home/jovyan"}

    def user(self):
        return "jovyan"

    def cwd(self):
        return "/home/jovyan"


def test_the_checker_checks_the_contract_it_says_it_does() -> None:
    contract = SANDBOX_CONTRACT_V1
    assert doctor.CONTRACT == {
        "version": contract.version,
        "architecture": contract.architecture,
        "os_family": contract.os_family,
        "user": contract.user,
        "uid": contract.uid,
        "gid": contract.gid,
        "home": contract.home,
        "workdir": contract.workdir,
        "reserved_path": contract.reserved_path,
        "python_executables": list(contract.python_executables),
        "kernel_packages": list(contract.kernel_packages),
        "locale": contract.locale,
        "timezone": contract.timezone,
    }


def test_a_compliant_sandbox_passes_every_row() -> None:
    report = doctor.check(CompliantHost())
    assert report["ok"] is True and report["failed"] is None
    assert [row["id"] for row in report["rows"]] == list(doctor.ROW_IDS)
    assert report["paths"] == {
        "writable": ["/home/datalayer/content"],
        "readOnly": ["/opt/datalayer"],
    }


def test_the_image_before_the_move_fails_exactly_what_the_move_and_the_contract_layer_fix() -> None:
    """E0-11 fixes user, home and workdir; the E1-05 contract layer adds /opt/datalayer."""
    report = doctor.check(JupyterPython011())
    failing = [row["id"] for row in report["rows"] if not row["ok"]]
    assert failing == ["user", "home", "workdir", "reserved_path"]
    assert report["failed"] == "user"


def test_a_writable_reserved_path_fails() -> None:
    class Writable(CompliantHost):
        writable: ClassVar[set[str]] = {"/home/datalayer/content", "/opt/datalayer"}

    report = doctor.check(Writable())
    assert report["failed"] == "reserved_path"
    assert (
        "writable" in next(row for row in report["rows"] if row["id"] == "reserved_path")["message"]
    )


def test_unreaped_orphans_fail_the_init_row() -> None:
    class Zombies(CompliantHost):
        def orphan_zombies(self):
            return 2

    assert doctor.check(Zombies())["failed"] == "init"


@pytest.mark.parametrize("host", [CompliantHost(), JupyterPython011()])
def test_the_report_is_pinned_by_its_schema(host: doctor.Host) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(doctor.check(host), schema)


def test_the_exit_code_and_stderr_name_the_first_failure() -> None:
    out, err = io.StringIO(), io.StringIO()
    assert doctor.main(["doctor", "--json"], host=JupyterPython011(), stdout=out, stderr=err) == 1
    assert json.loads(out.getvalue())["failed"] == "user"
    assert (
        err.getvalue() == "datalayer-sandbox doctor: user failed: runs as jovyan, not datalayer\n"
    )

    out, err = io.StringIO(), io.StringIO()
    assert doctor.main(["doctor"], host=CompliantHost(), stdout=out, stderr=err) == 0
    assert out.getvalue().splitlines()[0].startswith("ok   architecture")
    assert err.getvalue() == ""


def test_without_a_command_it_explains_itself() -> None:
    err = io.StringIO()
    assert doctor.main([], stderr=err) == 2
    assert "doctor" in err.getvalue()


def test_the_checker_is_standard_library_only_and_python_3_8() -> None:
    source = Path(doctor.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 8))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported


def test_the_zipapp_runs_on_its_own(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    target = build_zipapp(tmp_path / "datalayer-sandbox")
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(target), "doctor", "--json"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    report = json.loads(completed.stdout)
    jsonschema.validate(report, json.loads(SCHEMA.read_text(encoding="utf-8")))
    assert completed.returncode == (0 if report["ok"] else 1)
    if not report["ok"]:
        assert f"{report['failed']} failed" in completed.stderr
    version = subprocess.run(  # noqa: S603
        [sys.executable, str(target), "--version"], capture_output=True, text=True, timeout=60
    )
    assert version.stdout.strip() == "datalayer-sandbox 1"


def broken_host(row: str) -> doctor.Host:
    """A compliant host with exactly one row of the contract broken."""
    host = CompliantHost()
    breakers = {
        "architecture": lambda: setattr(host, "machine", lambda: "aarch64"),
        "os": lambda: setattr(host, "os_release", lambda: {"ID": "alpine"}),
        "user": lambda: setattr(host, "user", lambda: "root"),
        "uid": lambda: setattr(host, "uid", lambda: 0),
        "gid": lambda: setattr(host, "gid", lambda: 0),
        "home": lambda: setattr(host, "environment", {"HOME": "/root", "LANG": "C.UTF-8"}),
        "workdir": lambda: setattr(host, "writable", set()),
        "python": lambda: setattr(
            host, "which", lambda name: None if name == "python3" else "/bin/" + name
        ),
        "pip": lambda: setattr(
            host, "which", lambda name: None if name == "pip" else "/bin/" + name
        ),
        "kernel": lambda: setattr(
            host, "package_version", lambda name: "6.29.5" if name == "ipykernel" else None
        ),
        "reserved_path": lambda: setattr(
            host, "directories", {"/home/datalayer", "/home/datalayer/content"}
        ),
        "locale": lambda: setattr(
            host, "environment", {"HOME": "/home/datalayer", "LANG": "en_US.UTF-8"}
        ),
        "timezone": lambda: setattr(host, "timezone", lambda: "Europe/Paris"),
        "ca_certificates": lambda: setattr(host, "ca_bundle", lambda: None),
        "init": lambda: setattr(host, "orphan_zombies", lambda: None),
    }
    breakers[row]()
    return host


@pytest.mark.parametrize("row", doctor.ROW_IDS)
def test_each_row_of_the_contract_fails_on_its_own(row: str) -> None:
    """A passing and a failing fixture for every row: the failing one breaks that row alone."""
    report = doctor.check(broken_host(row))
    assert [item["id"] for item in report["rows"] if not item["ok"]] == [row]
    assert report["failed"] == row
    assert next(item for item in report["rows"] if item["id"] == row)["message"]
