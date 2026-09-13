# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The E2B builder: a template from the lock, and the build id kept (E2-03).

Nothing here reaches E2B: `Template`/`TemplateBuilder` are doubles that record
what they were asked, the way `datalayer.py`'s own tests record `buildctl`'s
argv without a daemon. What is checked is what the build chain *is* — starting
from `code-interpreter-v1`, the doctor and the lock copied in, the packages
installed as root, `postInstall` as `datalayer`, and the reference kept
afterwards.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_sandboxes.environments.adapters.e2b import CODE_INTERPRETER_BASE_TEMPLATE, Builder
from code_sandboxes.environments.builders import ArtifactReference, BuildRequest
from code_sandboxes.environments.errors import BUILD_FAILED, EnvironmentsError
from code_sandboxes.environments.spec import parse_environment

OWNER = "01k0wner000000000000000000"
BASE = "environments/base/python-cpu@sha256:" + "bb" * 32
LOCK = (
    "# Resolved by Datalayer (PLAN_ENV.md D-9). Do not edit: a change makes a new version.\n"
    "# python: 3.13\n"
    "# datalayer-protected: ipykernel==7.3.0\n"
    "geopandas==1.1.1 \\\n    --hash=sha256:" + "cd" * 32 + "\n"
)
LOCK_DIGEST = "sha256:" + "dd" * 32


class Call:
    """One method call a `FakeTemplateBuilder` recorded: its name and arguments."""

    def __init__(self, name: str, args: tuple, kwargs: dict) -> None:
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Call({self.name!r}, {self.args!r}, {self.kwargs!r})"


class FakeTemplateBuilder:
    """A `TemplateBuilder` chain: every call returns `self` and is recorded."""

    def __init__(self, calls: list[Call]) -> None:
        self._calls = calls

    def _record(self, name: str, *args, **kwargs) -> FakeTemplateBuilder:
        self._calls.append(Call(name, args, kwargs))
        return self

    def from_template(self, *a, **k):
        return self._record("from_template", *a, **k)

    def copy(self, *a, **k):
        return self._record("copy", *a, **k)

    def run_cmd(self, *a, **k):
        return self._record("run_cmd", *a, **k)

    def set_envs(self, *a, **k):
        return self._record("set_envs", *a, **k)

    def set_user(self, *a, **k):
        return self._record("set_user", *a, **k)

    def set_workdir(self, *a, **k):
        return self._record("set_workdir", *a, **k)


class BuildInfo:
    def __init__(self, template_id: str, build_id: str, name: str) -> None:
        self.template_id = template_id
        self.build_id = build_id
        self.name = name


class Tag:
    def __init__(self, name: str, build_id: str, created_at: str = "") -> None:
        self.name = name
        self.build_id = build_id
        self.created_at = created_at


class FakeTemplate:
    """A `Template` class double: constructs chains and answers `build`/`get_tags`/`exists`."""

    def __init__(
        self,
        *,
        build_id: str = "bld-1",
        template_id: str = "tpl-1",
        build_error: Exception | None = None,
        tags: list[Tag] | None = None,
        existing_names: set[str] | None = None,
    ) -> None:
        self.calls: list[Call] = []
        self.build_calls: list[Call] = []
        self._build_id = build_id
        self._template_id = template_id
        self._build_error = build_error
        self._tags = tags if tags is not None else []
        self._existing_names = existing_names if existing_names is not None else {template_id}

    def __call__(self, *, file_context_path: str | None = None) -> FakeTemplateBuilder:
        self.file_context_path = file_context_path
        return FakeTemplateBuilder(self.calls)

    def build(self, chain, name, *, tags=None, on_build_logs=None):
        self.build_calls.append(Call("build", (chain, name), {"tags": tags}))
        if on_build_logs is not None:
            on_build_logs(type("Entry", (), {"level": "info", "message": "solved"})())
        if self._build_error is not None:
            raise self._build_error
        return BuildInfo(self._template_id, self._build_id, name)

    def get_tags(self, template_id_or_name):
        return self._tags

    def exists(self, template_id_or_name):
        return template_id_or_name in self._existing_names


def a_request(**changes) -> BuildRequest:
    spec = {
        "apiVersion": "environments.datalayer.io/v1alpha1",
        "kind": "Environment",
        "metadata": {"name": "geospatial-analysis", "title": "Geospatial analysis"},
        "spec": {
            "language": {"name": "python", "version": "3.13"},
            "base": {"ref": "datalayer/python-cpu", "channel": "2026.09"},
            "packages": {
                "python": {"manager": "uv", "dependencies": ["geopandas==1.1.1"]},
            },
            "env": {"GDAL_DATA": "/usr/share/gdal"},
            "commands": {"postInstall": ["python -c 'import geopandas'"]},
            "resources": {"sizeClass": "medium"},
            "compatibility": {"variants": {"required": ["e2b"]}},
            **changes.pop("spec", {}),
        },
    }
    fields = {
        "environment_uid": "01k0env0000000000000000000",
        "version": 3,
        "build_uid": "bld-1",
        "owner_uid": OWNER,
        "variant": "e2b",
        "environment": parse_environment(spec),
        "lock_text": LOCK,
        "lock_digest": LOCK_DIGEST,
        "resolved_base": BASE,
        "region": None,
        "size_class": "medium",
    }
    fields.update(changes)
    return BuildRequest(**fields)


def a_builder(template: FakeTemplate | None = None, **changes) -> Builder:
    options = {
        "template_cls": lambda: (template or FakeTemplate()),
        "zipapp_builder": lambda target: Path(target).write_text("#!/usr/bin/env python3\n"),
        "team": "acme-team",
    }
    options.update(changes)
    return Builder(**options)


class TestBuildingATemplate:
    def test_it_starts_from_code_interpreters_own_template(self) -> None:
        fake = FakeTemplate()
        a_builder(fake).build(a_request())
        assert fake.calls[0].name == "from_template"
        assert fake.calls[0].args == (CODE_INTERPRETER_BASE_TEMPLATE,)

    def test_the_doctor_and_lock_are_copied_in(self) -> None:
        fake = FakeTemplate()
        a_builder(fake).build(a_request())
        copies = [call for call in fake.calls if call.name == "copy"]
        destinations = [call.args[1] for call in copies]
        assert "/opt/datalayer/bin/datalayer-sandbox" in destinations
        assert "/opt/datalayer/wheelhouse" in destinations
        assert "/opt/datalayer/lock.txt" in destinations
        doctor_copy = next(call for call in copies if call.args[1].endswith("datalayer-sandbox"))
        assert doctor_copy.kwargs["mode"] == 0o755

    def test_packages_install_as_root_postinstall_as_datalayer(self) -> None:
        """`code-interpreter-v1` sets its own persistent default user, so
        root-needing steps name `user="root"` explicitly (found live,
        2026-09-13) rather than relying on no override meaning root.
        `postInstall` names no user at all: it inherits the `datalayer`
        default `set_user` switched to right after the user was created —
        also found live, 2026-09-13: a user just created mid-build is not
        one `run_cmd(user=...)` can name yet, only `set_user` can."""
        fake = FakeTemplate()
        a_builder(fake).build(a_request())
        install = next(
            call for call in fake.calls if call.name == "run_cmd" and "uv pip sync" in call.args[0]
        )
        assert install.kwargs["user"] == "root"
        post_install = next(
            call for call in fake.calls if call.name == "run_cmd" and "geopandas" in call.args[0]
        )
        assert "user" not in post_install.kwargs

    def test_the_datalayer_user_is_created_as_root_then_made_the_default(self) -> None:
        """`useradd` as root, then `set_user("datalayer")` names it as the
        default for every later step — found live, 2026-09-13, to be the one
        ordering that runs the whole chain through cleanly: `set_user`
        before a real account exists left the account unable to exec
        anything at all ("/bin/sh: permission denied")."""
        fake = FakeTemplate()
        a_builder(fake).build(a_request())
        setup = next(
            call for call in fake.calls if call.name == "run_cmd" and "useradd" in call.args[0]
        )
        assert setup.kwargs["user"] == "root"
        assert "groupadd -g 100 datalayer" in setup.args[0]
        assert "-u 1000 -g 100 -m -d /home/datalayer" in setup.args[0]
        set_user_calls = [call for call in fake.calls if call.name == "set_user"]
        assert set_user_calls[0].args == ("datalayer",)
        # useradd comes before set_user, which comes before any copy/run_cmd
        # that must run as root overrides it back per call.
        assert fake.calls.index(setup) < fake.calls.index(set_user_calls[0])

    def test_the_doctor_runs_at_build_time(self) -> None:
        fake = FakeTemplate()
        a_builder(fake).build(a_request())
        doctor_run = next(
            call
            for call in fake.calls
            if call.name == "run_cmd" and "datalayer-sandbox doctor --json" in call.args[0]
        )
        # No user= override: it inherits the datalayer default set_user
        # already switched to, the same as postInstall.
        assert "user" not in doctor_run.kwargs

    def test_user_and_workdir_are_set_last(self) -> None:
        fake = FakeTemplate()
        a_builder(fake).build(a_request())
        names = [call.name for call in fake.calls]
        assert names[-2:] == ["set_user", "set_workdir"]
        assert fake.calls[-2].args == ("datalayer",)
        assert fake.calls[-1].args == ("/home/datalayer/content",)

    def test_env_is_set_when_the_spec_has_any(self) -> None:
        fake = FakeTemplate()
        a_builder(fake).build(a_request())
        set_envs = next(call for call in fake.calls if call.name == "set_envs")
        assert set_envs.args == ({"GDAL_DATA": "/usr/share/gdal"},)

    def test_no_env_call_when_the_spec_sets_none(self) -> None:
        fake = FakeTemplate()
        a_builder(fake).build(a_request(spec={"env": {}}))
        assert not any(call.name == "set_envs" for call in fake.calls)

    def test_the_template_is_named_and_tagged_by_version_and_build(self) -> None:
        fake = FakeTemplate()
        a_builder(fake).build(a_request())
        assert fake.build_calls[0].args[1] == "dl-geospatial-analysis-v3"
        assert fake.build_calls[0].kwargs["tags"] == ["v3-bld-1"]

    def test_build_logs_reach_the_workflows_log(self) -> None:
        seen: list[str] = []
        fake = FakeTemplate()
        a_builder(fake, log=seen.append).build(a_request())
        assert any("solved" in line for line in seen)

    def test_the_artifact_is_the_build_id_never_a_name_alone(self) -> None:
        fake = FakeTemplate(build_id="bld-42", template_id="tpl-9")
        artifact = a_builder(fake).build(a_request())
        assert artifact.variant == "e2b"
        assert artifact.immutable_reference == "acme-team/dl-geospatial-analysis-v3:bld-42"
        assert artifact.provider_artifact_id == "bld-42"
        assert artifact.mutable_alias == "tpl-9:dl-geospatial-analysis-v3"
        assert artifact.contract_version == "sandbox-contract/v1"

    def test_with_no_team_the_reference_has_no_namespace(self) -> None:
        fake = FakeTemplate()
        artifact = a_builder(fake, team=None).build(a_request())
        assert artifact.immutable_reference == "dl-geospatial-analysis-v3:bld-1"

    def test_a_build_failure_is_reported_with_its_log(self) -> None:
        fake = FakeTemplate(build_error=RuntimeError("quota exceeded"))
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(fake).build(a_request())
        assert raised.value.code is BUILD_FAILED
        assert "quota exceeded" in raised.value.message
        assert raised.value.detail["log"]


class TestReadingTheRegistry:
    def _an_artifact(self, template_id="tpl-1", build_id="bld-1") -> ArtifactReference:
        return ArtifactReference(
            variant="e2b",
            immutable_reference=f"acme-team/dl-geo-v3:{build_id}",
            provider_artifact_id=build_id,
            mutable_alias=f"{template_id}:dl-geo-v3",
            contract_version="sandbox-contract/v1",
        )

    def test_inspect_reads_the_matching_tag(self) -> None:
        fake = FakeTemplate(tags=[Tag("v3-bld-1", "bld-1", created_at="2026-09-13T00:00:00Z")])
        metadata = a_builder(fake).inspect(self._an_artifact())
        assert metadata.created_at == "2026-09-13T00:00:00Z"
        assert metadata.labels == {"tag": "v3-bld-1"}

    def test_inspect_refuses_a_build_id_with_no_matching_tag(self) -> None:
        fake = FakeTemplate(tags=[])
        with pytest.raises(EnvironmentsError):
            a_builder(fake).inspect(self._an_artifact())

    def test_exists_is_true_only_with_a_matching_tag(self) -> None:
        fake = FakeTemplate(tags=[Tag("v3-bld-1", "bld-1")], existing_names={"tpl-1"})
        assert a_builder(fake).exists(self._an_artifact()) is True
        assert a_builder(fake).exists(self._an_artifact(build_id="bld-2")) is False

    def test_exists_is_false_once_the_template_itself_is_gone(self) -> None:
        fake = FakeTemplate(tags=[Tag("v3-bld-1", "bld-1")], existing_names=set())
        assert a_builder(fake).exists(self._an_artifact()) is False
