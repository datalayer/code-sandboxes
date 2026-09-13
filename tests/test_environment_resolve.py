# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Resolution: one lock, Datalayer's pins over the user's, and uv's refusals read (E1-04).

The refusals below are **recorded**: each one is what `uv pip compile`
(0.12.11) actually wrote for that input, box drawing and wrapping included, so
the parser is tested against the resolver's own words rather than against a
paraphrase of them that cannot go out of date.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

import httpx
import pytest

from code_sandboxes.environments.bases import ApprovedBase, BaseChannelUnpublishedError
from code_sandboxes.environments.errors import EnvironmentsError
from code_sandboxes.environments.resolve import (
    APT_PIN_PREFIX,
    LOCK_FORMAT,
    LocalResolveRunner,
    MergedRequirements,
    ResolveOutcome,
    ResolveRequest,
    apt_pins,
    lock_document,
    locked_versions,
    merge_requirements,
    parse_resolver_failure,
    protected_pins,
    resolve_bases,
    resolve_environment,
    resolve_image_base,
)

# -- What uv wrote ------------------------------------------------------------

#: `requests==2.31.0` with `urllib3==1.21`: the conflict is derived, not stated.
UV_CONFLICT = """\
  × No solution found when resolving dependencies:
  ╰─▶ Because requests==2.31.0 depends on urllib3>=1.21.1,<3 and you require
      requests==2.31.0, we can conclude that you require urllib3>=1.21.1,<3.
      And because you require urllib3==1.21, we can conclude that your
      requirements are unsatisfiable.
"""

#: A package no index has.
UV_MISSING = """\
  × No solution found when resolving dependencies:
  ╰─▶ Because datalayer-no-such-package-xyzzy was not found in the package
      registry and you require datalayer-no-such-package-xyzzy==1.0.0, we can
      conclude that your requirements are unsatisfiable.
"""

#: `ipykernel==6.0.0` against Datalayer's `ipykernel==7.3.0` constraint.
UV_PROTECTED = """\
  × No solution found when resolving dependencies:
  ╰─▶ Because you require ipykernel==6.0.0 and ipykernel==7.3.0, we can
      conclude that your requirements are unsatisfiable.
"""

#: Not a resolution failure at all.
UV_UNREACHABLE = """\
  × Failed to fetch: `https://pypi.org/simple/geopandas/`
  ╰─▶ Request failed after 3 retries
"""

BASES = {
    "datalayer/python-cpu": ApprovedBase(
        ref="datalayer/python-cpu",
        python_versions=("3.13",),
        channels={
            "2026.09": {"datalayer": "sha256:" + "11" * 32, "modal": "sha256:" + "22" * 32},
            "2026.10": {},
        },
    )
}


def a_spec(**spec: object) -> dict[str, object]:
    """The section 4.1 example, with whatever this test changes about it."""
    packages: dict[str, object] = {
        "python": {"manager": "uv", "dependencies": ["geopandas==1.1.1", "rasterio==1.4.3"]}
    }
    return {
        "apiVersion": "environments.datalayer.io/v1alpha1",
        "kind": "Environment",
        "metadata": {"name": "geospatial-analysis", "title": "Geospatial analysis"},
        "spec": {
            "language": {"name": "python", "version": "3.13"},
            "base": {"ref": "datalayer/python-cpu", "channel": "2026.09"},
            "packages": packages,
            "resources": {"sizeClass": "medium"},
            "compatibility": {"variants": {"required": ["datalayer"], "optional": ["modal"]}},
            **spec,
        },
    }


class RecordedRunner:
    """A solve that answers what it was given, and remembers what it was asked."""

    name = "recorded"

    def __init__(self, outcome: ResolveOutcome | Exception) -> None:
        self._outcome = outcome
        self.request: ResolveRequest | None = None
        self.lines: list[str] = []

    def solve(self, request: ResolveRequest, log=None) -> ResolveOutcome:
        self.request = request
        if log is not None:
            log("solving")
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


A_LOCK = ResolveOutcome(
    lock_text=(
        "affine==3.0.1 \\\n    --hash=sha256:" + "ab" * 32 + "\n    # via rasterio\n"
        "geopandas==1.1.1 \\\n    --hash=sha256:" + "cd" * 32 + "\n"
        "rasterio==1.4.3 \\\n    --hash=sha256:" + "ef" * 32 + "\n"
    )
)


# -- Datalayer's pins over the user's ----------------------------------------


class TestTheProtectedPins:
    def test_they_are_the_kernel_stack_at_one_version_each(self) -> None:
        pins = {pin.name: pin for pin in protected_pins()}
        assert set(pins) == {
            "ipykernel",
            "jupyter-client",
            "jupyter-server",
            "jupyter-server-nbmodel",
            "datalayer",
        }
        assert all(pin.version for pin in pins.values())
        assert pins["jupyter-server"].version == "2.21.0+datalayer.1"

    def test_a_requirement_that_agrees_with_a_pin_is_dropped_for_it(self) -> None:
        # The fork satisfies `>=2.19`, which is what jupyterlab asks for, so
        # the user's own line is redundant rather than wrong — it is dropped,
        # and Datalayer's own pin is what ends up in the requirements instead
        # (below), the same as for every protected package, asked for or not.
        merged = merge_requirements(["jupyter-server>=2.19", "geopandas==1.1.1"])
        assert "geopandas==1.1.1" in merged.requirements
        assert "jupyter-server>=2.19" not in merged.requirements
        assert "jupyter-server==2.21.0+datalayer.1" in merged.requirements
        assert "jupyter-server==2.21.0+datalayer.1" in merged.constraints
        assert any("jupyter-server" in note for note in merged.notes)

    def test_every_protected_pin_is_a_requirement_whether_or_not_the_spec_asked(self) -> None:
        """A `uv --constraint` only bounds what is already in the graph.

        It never pulls a package in — found live on 2026-09-12, where a spec
        with no kernel dependency of its own locked one package, and
        `uv pip sync` then removed the base image's own ipykernel and
        jupyter_client for not being in that lock. The contract's whole point
        is that every artifact can start a kernel, so every pin is forced in.
        """
        merged = merge_requirements(["six==1.16.0"])
        assert "six==1.16.0" in merged.requirements
        for pin in protected_pins():
            assert pin.requirement in merged.requirements

    def test_a_requirement_that_contradicts_a_pin_is_refused_with_the_range(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            merge_requirements(["ipykernel==6.0.0"])
        error = raised.value
        assert error.code.code == "DL_ENV_PROTECTED_PACKAGE"
        assert error.detail["package"] == "ipykernel"
        assert error.detail["requested"] == "==6.0.0"
        assert error.detail["supported"] == "==7.3.0"
        assert error.detail["field"] == "spec.packages.python.dependencies"

    def test_a_constraint_of_the_users_is_read_the_same_way(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            merge_requirements([], ["datalayer<1.0"])
        assert raised.value.detail["field"] == "spec.packages.python.constraints"

    def test_the_pins_are_passed_to_the_solve_last_so_they_win(self) -> None:
        merged = merge_requirements(["geopandas==1.1.1"], ["numpy>=2"])
        assert merged.constraints[0] == "numpy>=2"
        assert merged.constraints[-1].startswith("datalayer==")

    def test_an_unreadable_requirement_is_the_specs_fault(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            merge_requirements(["not a requirement at all"])
        assert raised.value.code.code == "DL_ENV_SPEC_INVALID"


# -- Reading uv ---------------------------------------------------------------


class TestUvsRefusals:
    def test_a_conflict_names_the_two_requirements_that_cannot_hold(self) -> None:
        error = parse_resolver_failure(UV_CONFLICT)
        assert error.code.code == "DL_ENV_RESOLVE_CONFLICT"
        # The pair uv derived, not the pair the user typed: `requests==2.31.0`
        # is fine, and it is urllib3 that cannot be both.
        assert error.detail["conflict"] == ["urllib3>=1.21.1,<3", "urllib3==1.21"]

    def test_a_package_no_index_has_is_named(self) -> None:
        error = parse_resolver_failure(UV_MISSING)
        assert error.code.code == "DL_ENV_PACKAGE_NOT_FOUND"
        assert error.detail["package"] == "datalayer-no-such-package-xyzzy"

    def test_a_protected_package_is_that_code_and_not_a_conflict(self) -> None:
        error = parse_resolver_failure(UV_PROTECTED)
        assert error.code.code == "DL_ENV_PROTECTED_PACKAGE"
        assert error.detail == {
            **{"output": error.detail["output"]},
            "package": "ipykernel",
            "requested": "==6.0.0",
            "supported": "==7.3.0",
        }

    def test_a_pin_nobody_serves_reads_as_protected_rather_than_missing(self) -> None:
        # The fork's local version is in no index: it is the copy in the base
        # image, and a user requirement is what put it in the graph.
        recorded = UV_MISSING.replace("datalayer-no-such-package-xyzzy", "jupyter-server")
        error = parse_resolver_failure(recorded)
        assert error.code.code == "DL_ENV_PROTECTED_PACKAGE"
        assert error.detail["package"] == "jupyter-server"

    def test_a_failure_that_is_not_about_the_version_is_retryable(self) -> None:
        error = parse_resolver_failure(UV_UNREACHABLE)
        assert error.code.code == "DL_ENV_PROVIDER_ERROR"
        assert error.code.retry.value != "no"

    def test_every_recorded_refusal_carries_the_output_it_was_read_from(self) -> None:
        for recorded in (UV_CONFLICT, UV_MISSING, UV_PROTECTED, UV_UNREACHABLE):
            error = parse_resolver_failure(recorded)
            assert (
                "No solution found" in error.detail["output"]
                or "Failed to fetch" in error.detail["output"]
            )
            # One line, so a log reads it without the box drawing.
            assert "╰" not in error.detail["output"]


# -- The lock -----------------------------------------------------------------


class TestTheLock:
    def test_it_is_a_requirements_file_with_what_else_is_installed_in_comments(self) -> None:
        outcome = ResolveOutcome(
            lock_text=A_LOCK.lock_text,
            apt_pins={"gdal-bin": "3.8.4+dfsg-3build2"},
            apt_source="https://snapshot.ubuntu.com/ubuntu/20260901T000000Z",
        )
        document = lock_document(
            outcome,
            python_version="3.13",
            base_reference="environments/base/python-cpu@sha256:" + "11" * 32,
            merged=merge_requirements(["geopandas==1.1.1"]),
            resolved_at=datetime(2026, 9, 12, 8, 30, tzinfo=timezone.utc),
        )
        assert document["format"] == LOCK_FORMAT
        assert document["digest"].startswith("sha256:")
        assert document["python_version"] == "3.13"
        assert document["package_count"] == 3
        content = document["content"]
        assert f"{APT_PIN_PREFIX}gdal-bin=3.8.4+dfsg-3build2" in content
        assert "# datalayer-protected: ipykernel==7.3.0" in content
        assert "# resolved-at: 2026-09-12T08:30:00+00:00" in content
        # Still a requirements file: what a reader of one sees is the packages.
        assert locked_versions(content) == {
            "affine": "3.0.1",
            "geopandas": "1.1.1",
            "rasterio": "1.4.3",
        }

    def test_the_same_lock_digests_the_same_and_a_changed_one_does_not(self) -> None:
        arguments = {
            "python_version": "3.13",
            "base_reference": "environments/base/python-cpu@sha256:" + "11" * 32,
            "merged": MergedRequirements((), (), ()),
            "resolved_at": datetime(2026, 9, 12, tzinfo=timezone.utc),
        }
        first = lock_document(A_LOCK, **arguments)
        again = lock_document(A_LOCK, **arguments)
        assert first["digest"] == again["digest"]
        moved = lock_document(
            ResolveOutcome(lock_text=A_LOCK.lock_text.replace("1.1.1", "1.1.2")), **arguments
        )
        assert moved["digest"] != first["digest"]

    def test_an_apt_simulation_pins_every_package_it_would_install(self) -> None:
        # What `apt-get install --simulate` writes, transitive packages too.
        simulation = (
            "NOTE: This is only a simulation!\n"
            "Inst libgdal34 (3.8.4+dfsg-3build2 Ubuntu:24.04/noble [amd64])\n"
            "Inst gdal-bin (3.8.4+dfsg-3build2 Ubuntu:24.04/noble [amd64])\n"
            "Conf gdal-bin (3.8.4+dfsg-3build2 Ubuntu:24.04/noble [amd64])\n"
        )
        assert apt_pins(simulation) == {
            "libgdal34": "3.8.4+dfsg-3build2",
            "gdal-bin": "3.8.4+dfsg-3build2",
        }


# -- The bases ----------------------------------------------------------------


class TestTheBases:
    def test_each_variant_gets_the_channels_digest_as_a_reference(self) -> None:
        from code_sandboxes.environments.spec import parse_environment

        resolved = resolve_bases(parse_environment(a_spec()), ["datalayer", "modal"], BASES)
        assert resolved == {
            "datalayer": "environments/base/python-cpu@sha256:" + "11" * 32,
            "modal": "environments/base/python-cpu@sha256:" + "22" * 32,
        }

    def test_a_registry_qualifies_every_base_so_from_resolves_anywhere_but_docker_hub(
        self,
    ) -> None:
        """A bare `<repository>@sha256:…` `FROM`s Docker Hub, which is not where it lives.

        Found live on 2026-09-12: the first real BuildKit solve against a
        published base failed `pull access denied` from `docker.io`, because
        nothing had ever qualified the reference with the registry it is
        actually in. Runtimes' own `resolvedBases` validation already
        documents `<registry>/<repository>@sha256:<hex>` as the shape it
        stores; this is the other end of that contract.
        """
        from code_sandboxes.environments.spec import parse_environment

        resolved = resolve_bases(
            parse_environment(a_spec()),
            ["datalayer", "modal"],
            BASES,
            registry="123456789012.dkr.ecr.us-east-1.amazonaws.com",
        )
        assert resolved == {
            "datalayer": "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
            "environments/base/python-cpu@sha256:" + "11" * 32,
            "modal": "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
            "environments/base/python-cpu@sha256:" + "22" * 32,
        }

    def test_resolve_environment_qualifies_the_base_from_the_credentials_registry(
        self,
    ) -> None:
        class Credential:
            registry = "123456789012.dkr.ecr.us-east-1.amazonaws.com"

        document = resolve_environment(
            spec=a_spec(),
            variants=["datalayer"],
            credential=Credential(),
            runner=RecordedRunner(A_LOCK),
            bases=BASES,
        )
        assert document["resolved_bases"]["datalayer"] == (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
            "environments/base/python-cpu@sha256:" + "11" * 32
        )

    def test_a_channel_nobody_published_refuses_by_name(self) -> None:
        spec = a_spec(base={"ref": "datalayer/python-cpu", "channel": "2026.10"})
        with pytest.raises(BaseChannelUnpublishedError) as raised:
            resolve_environment(
                spec=spec, variants=["datalayer"], runner=RecordedRunner(A_LOCK), bases=BASES
            )
        assert raised.value.detail["reason"] == "base_channel_unpublished"


# -- The seam -----------------------------------------------------------------


class TestResolvingAVersion:
    def test_it_answers_what_the_workflow_stores(self) -> None:
        runner = RecordedRunner(A_LOCK)
        answer = resolve_environment(
            spec=a_spec(), variants=["datalayer", "modal"], runner=runner, bases=BASES
        )
        assert set(answer) == {
            "digest",
            "format",
            "content",
            "python_version",
            "package_count",
            "resolved_bases",
        }
        assert answer["resolved_bases"]["modal"].endswith("22" * 32)
        assert answer["package_count"] == 3

    def test_the_solve_runs_in_the_datalayer_base_with_the_merged_requirements(self) -> None:
        runner = RecordedRunner(A_LOCK)
        resolve_environment(
            spec=a_spec(), variants=["modal", "datalayer"], runner=runner, bases=BASES
        )
        request = runner.request
        assert request is not None
        # The interpreter that resolves is the one the artifact will have (D-9).
        assert request.base_reference.endswith("11" * 32)
        assert request.python_version == "3.13"
        assert "geopandas==1.1.1" in request.requirements
        assert "rasterio==1.4.3" in request.requirements
        # Every protected pin is forced in too, whether or not the spec asked.
        assert "ipykernel==7.3.0" in request.requirements
        assert "ipykernel==7.3.0" in request.constraints
        assert request.indexes == ("https://pypi.org/simple",)

    def test_the_lock_the_solve_answered_is_the_lock_that_is_stored(self) -> None:
        answer = resolve_environment(
            spec=a_spec(), variants=["datalayer"], runner=RecordedRunner(A_LOCK), bases=BASES
        )
        assert "geopandas==1.1.1" in answer["content"]
        assert locked_versions(answer["content"])["rasterio"] == "1.4.3"

    def test_a_refusal_from_the_solve_is_the_refusal_the_caller_gets(self) -> None:
        runner = RecordedRunner(parse_resolver_failure(UV_CONFLICT))
        with pytest.raises(EnvironmentsError) as raised:
            resolve_environment(spec=a_spec(), variants=["datalayer"], runner=runner, bases=BASES)
        assert raised.value.code.code == "DL_ENV_RESOLVE_CONFLICT"

    def test_a_form_that_is_not_packages_is_refused_by_name(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            resolve_environment(
                spec=a_spec(build={"source": "dockerfile"}),
                variants=["datalayer"],
                runner=RecordedRunner(A_LOCK),
                bases=BASES,
            )
        assert raised.value.code.code == "DL_ENV_CAPABILITY_UNSUPPORTED"
        assert raised.value.detail["source"] == "dockerfile"

    def test_conda_waits_for_its_own_solver(self) -> None:
        spec = a_spec(
            packages={"python": {"manager": "conda", "dependencies": ["geopandas=1.1.1"]}}
        )
        with pytest.raises(EnvironmentsError) as raised:
            resolve_environment(
                spec=spec, variants=["datalayer"], runner=RecordedRunner(A_LOCK), bases=BASES
            )
        assert raised.value.code.code == "DL_ENV_CAPABILITY_UNSUPPORTED"
        assert raised.value.detail["manager"] == "conda"

    def test_the_credentials_registry_auth_reaches_the_runner(self) -> None:
        class Credential:
            def registry_auth(self) -> dict[str, str]:
                return {"AWS_SESSION_TOKEN": "opaque"}

        runner = RecordedRunner(A_LOCK)
        resolve_environment(
            spec=a_spec(),
            variants=["datalayer"],
            credential=Credential(),
            runner=runner,
            bases=BASES,
        )
        assert runner.request is not None
        assert runner.request.registry_auth == {"AWS_SESSION_TOKEN": "opaque"}


# -- Dependency files (E3-01) --------------------------------------------------


def a_dependency_file_spec(**dependency_file: object) -> dict[str, object]:
    """A `requirements` source, with whatever this test changes about it."""
    return a_spec(
        packages={},
        build={
            "source": "dependencyFile",
            "dependencyFile": {
                "sourceFormat": "requirements",
                "content": "geopandas==1.1.1\nrasterio==1.4.3\n",
                **dependency_file,
            },
        },
    )


class TestARequirementsFile:
    def test_it_resolves_the_way_packages_does(self) -> None:
        """Same pipeline, same merge, same lock — just a different ingredient."""
        runner = RecordedRunner(A_LOCK)
        resolve_environment(
            spec=a_dependency_file_spec(), variants=["datalayer"], runner=runner, bases=BASES
        )
        request = runner.request
        assert request is not None
        assert "geopandas==1.1.1" in request.requirements
        assert "rasterio==1.4.3" in request.requirements
        # Every protected pin is still forced in, the same as for `packages`.
        assert "ipykernel==7.3.0" in request.requirements

    def test_comments_and_blank_lines_are_not_requirements(self) -> None:
        runner = RecordedRunner(A_LOCK)
        resolve_environment(
            spec=a_dependency_file_spec(
                content="# a comment\n\ngeopandas==1.1.1  # inline, too\n\nrasterio==1.4.3\n"
            ),
            variants=["datalayer"],
            runner=runner,
            bases=BASES,
        )
        assert runner.request is not None
        assert sorted(runner.request.requirements[:2]) == ["geopandas==1.1.1", "rasterio==1.4.3"]

    def test_a_pip_option_line_is_not_a_requirement_either(self) -> None:
        """Indexes come from `spec.packages.python.indexes`, not from the file."""
        runner = RecordedRunner(A_LOCK)
        resolve_environment(
            spec=a_dependency_file_spec(
                content="-r other.txt\n--index-url https://example/simple\nsix==1.16.0\n"
            ),
            variants=["datalayer"],
            runner=runner,
            bases=BASES,
        )
        assert runner.request is not None
        assert runner.request.requirements[0] == "six==1.16.0"

    def test_the_same_list_produces_the_same_lock_as_packages(self) -> None:
        packages_spec = a_spec(
            packages={"python": {"dependencies": ["geopandas==1.1.1", "rasterio==1.4.3"]}}
        )
        from_packages = resolve_environment(
            spec=packages_spec, variants=["datalayer"], runner=RecordedRunner(A_LOCK), bases=BASES
        )
        from_file = resolve_environment(
            spec=a_dependency_file_spec(),
            variants=["datalayer"],
            runner=RecordedRunner(A_LOCK),
            bases=BASES,
        )
        assert from_packages["content"] == from_file["content"]
        assert from_packages["digest"] == from_file["digest"]


# -- A pyproject.toml and its uv.lock (E3-01) ----------------------------------


class FakeUv:
    """`uv lock --dry-run` and `uv export`, without a real `uv`.

    Each answer is `(returncode, stdout, stderr)`, consumed in call order —
    `--dry-run` first, `export` second, when the dry run found nothing to
    refuse.
    """

    def __init__(self, *answers: tuple[int, str, str]) -> None:
        self.answers = list(answers)
        self.calls: list[list[str]] = []

    def __call__(self, argv, **_kwargs):
        self.calls.append(list(argv))
        code, out, err = self.answers[min(len(self.calls), len(self.answers)) - 1]
        return subprocess.CompletedProcess(list(argv), code, out, err)


def a_pyproject_spec(**dependency_file: object) -> dict[str, object]:
    return a_spec(
        packages={},
        build={
            "source": "dependencyFile",
            "dependencyFile": {
                "sourceFormat": "pyproject",
                "content": '[project]\nname = "x"\ndependencies = ["six==1.16.0"]\n',
                "lockContent": "# a uv.lock\n",
                **dependency_file,
            },
        },
    )


EXPORTED = (
    "# This file was autogenerated by uv via the following command:\n"
    "#    uv export --locked --format requirements.txt\n"
    "six==1.16.0 \\\n    --hash=sha256:" + "aa" * 32 + "\n    # via x\n"
)


class TestAPyprojectFile:
    def test_a_current_lock_is_exported_rather_than_resolved(self) -> None:
        uv = FakeUv((0, "", "Resolved 2 packages in 1ms\n"), (0, EXPORTED, ""))
        answer = resolve_environment(
            spec=a_pyproject_spec(),
            variants=["datalayer"],
            bases=BASES,
            uv="/usr/bin/uv",
            pyproject_run=uv,
        )
        assert answer["content"] == EXPORTED
        assert answer["package_count"] == 1
        assert answer["python_version"] == "3.13"
        assert uv.calls[0][:2] == ["/usr/bin/uv", "lock"]
        assert uv.calls[1][:2] == ["/usr/bin/uv", "export"]

    def test_an_updated_package_is_refused_naming_both_versions(self) -> None:
        uv = FakeUv((0, "", "Resolved 1 package in 1ms\nUpdate six v1.16.0 -> v1.17.0\n"))
        with pytest.raises(EnvironmentsError) as raised:
            resolve_environment(
                spec=a_pyproject_spec(),
                variants=["datalayer"],
                bases=BASES,
                uv="/usr/bin/uv",
                pyproject_run=uv,
            )
        assert raised.value.code.code == "DL_ENV_RESOLVE_CONFLICT"
        assert "six" in raised.value.message
        assert "1.16.0" in raised.value.message
        assert "1.17.0" in raised.value.message
        assert raised.value.detail["package"] == "six"

    def test_a_package_the_lock_is_missing_is_refused(self) -> None:
        uv = FakeUv((0, "", "Resolved 2 packages in 1ms\nAdd wheel v0.48.0\n"))
        with pytest.raises(EnvironmentsError) as raised:
            resolve_environment(
                spec=a_pyproject_spec(),
                variants=["datalayer"],
                bases=BASES,
                uv="/usr/bin/uv",
                pyproject_run=uv,
            )
        assert raised.value.code.code == "DL_ENV_RESOLVE_CONFLICT"
        assert raised.value.detail["package"] == "wheel"

    def test_a_package_the_lock_no_longer_needs_is_refused(self) -> None:
        uv = FakeUv((0, "", "Resolved 0 packages in 1ms\nRemove six v1.16.0\n"))
        with pytest.raises(EnvironmentsError) as raised:
            resolve_environment(
                spec=a_pyproject_spec(),
                variants=["datalayer"],
                bases=BASES,
                uv="/usr/bin/uv",
                pyproject_run=uv,
            )
        assert raised.value.code.code == "DL_ENV_RESOLVE_CONFLICT"
        assert raised.value.detail["package"] == "six"

    def test_no_uv_refuses_by_name(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            resolve_environment(spec=a_pyproject_spec(), variants=["datalayer"], bases=BASES, uv="")
        assert raised.value.code.code == "DL_ENV_CAPABILITY_UNSUPPORTED"
        assert raised.value.detail["missing"] == "uv"

    def test_a_dry_run_that_fails_for_an_unrelated_reason_is_a_provider_error(self) -> None:
        uv = FakeUv((2, "", "error: TOML parse error at line 1\n"))
        with pytest.raises(EnvironmentsError) as raised:
            resolve_environment(
                spec=a_pyproject_spec(),
                variants=["datalayer"],
                bases=BASES,
                uv="/usr/bin/uv",
                pyproject_run=uv,
            )
        assert raised.value.code.code == "DL_ENV_PROVIDER_ERROR"

    def test_export_failing_after_a_clean_dry_run_is_a_provider_error(self) -> None:
        uv = FakeUv((0, "", "Resolved 1 package in 1ms\n"), (1, "", "error: something odd\n"))
        with pytest.raises(EnvironmentsError) as raised:
            resolve_environment(
                spec=a_pyproject_spec(),
                variants=["datalayer"],
                bases=BASES,
                uv="/usr/bin/uv",
                pyproject_run=uv,
            )
        assert raised.value.code.code == "DL_ENV_PROVIDER_ERROR"

    def test_a_hanging_check_is_a_provider_error(self) -> None:
        def hangs(*_args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="uv", timeout=kwargs.get("timeout", 60))

        with pytest.raises(EnvironmentsError) as raised:
            resolve_environment(
                spec=a_pyproject_spec(),
                variants=["datalayer"],
                bases=BASES,
                uv="/usr/bin/uv",
                pyproject_run=hangs,
            )
        assert raised.value.code.code == "DL_ENV_PROVIDER_ERROR"


# -- An imported image (E3-04) -------------------------------------------------


IMAGE_DIGEST = "sha256:" + "9" * 64


def an_image_spec(**image: object) -> dict[str, object]:
    """An `image` source, with whatever this test changes about it."""
    return a_spec(
        packages={},
        base={"ref": "n/a", "channel": "n/a"},
        build={
            "source": "image",
            "image": {"reference": "python:3.12-slim-bookworm", **image},
        },
    )


def anonymous_manifest_transport(digest: str = IMAGE_DIGEST) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Docker-Content-Digest": digest})

    return httpx.MockTransport(handler)


class TestAnImportedImage:
    def test_the_image_is_the_base_for_every_variant_asked(self) -> None:
        from code_sandboxes.environments.spec import parse_environment

        resolved = resolve_image_base(
            parse_environment(an_image_spec()),
            ["datalayer", "modal"],
            transport=anonymous_manifest_transport(),
        )
        assert resolved == {
            "datalayer": f"docker.io/library/python@{IMAGE_DIGEST}",
            "modal": f"docker.io/library/python@{IMAGE_DIGEST}",
        }

    def test_a_reference_already_pinned_by_digest_needs_no_network_call(self) -> None:
        from code_sandboxes.environments.spec import parse_environment

        def unreachable(request: httpx.Request) -> httpx.Response:
            raise AssertionError("a pinned reference must never be looked up")

        resolved = resolve_image_base(
            parse_environment(an_image_spec(reference=f"python@{IMAGE_DIGEST}")),
            ["datalayer"],
            transport=httpx.MockTransport(unreachable),
        )
        assert resolved == {"datalayer": f"docker.io/library/python@{IMAGE_DIGEST}"}

    def test_a_disallowed_registry_is_policy_denied(self) -> None:
        from code_sandboxes.environments.spec import parse_environment

        with pytest.raises(EnvironmentsError) as raised:
            resolve_image_base(
                parse_environment(an_image_spec(reference="evil.example.com/x:y")),
                ["datalayer"],
            )
        assert raised.value.code.code == "DL_ENV_POLICY_DENIED"

    def test_resolve_environment_uses_the_images_digest_as_the_base(self) -> None:
        runner = RecordedRunner(A_LOCK)
        document = resolve_environment(
            spec=an_image_spec(),
            variants=["datalayer"],
            runner=runner,
            image_transport=anonymous_manifest_transport(),
        )
        assert document["resolved_bases"] == {
            "datalayer": f"docker.io/library/python@{IMAGE_DIGEST}"
        }
        assert runner.request is not None
        assert runner.request.base_reference == f"docker.io/library/python@{IMAGE_DIGEST}"
        # The contract layer, the same as every other source: nothing about
        # an import skips Datalayer's own protected pins.
        assert "ipykernel==7.3.0" in runner.request.requirements

    def test_resolve_environment_asks_uv_to_be_bootstrapped_for_an_image_source(self) -> None:
        runner = RecordedRunner(A_LOCK)
        resolve_environment(
            spec=an_image_spec(),
            variants=["datalayer"],
            runner=runner,
            image_transport=anonymous_manifest_transport(),
        )
        assert runner.request is not None
        assert runner.request.bootstrap_uv is True

    def test_a_packages_source_never_bootstraps_uv(self) -> None:
        runner = RecordedRunner(A_LOCK)
        resolve_environment(spec=a_spec(), variants=["datalayer"], runner=runner, bases=BASES)
        assert runner.request is not None
        assert runner.request.bootstrap_uv is False

    def test_an_unlisted_source_still_refuses_the_way_it_always_has(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            resolve_environment(
                spec=a_spec(build={"source": "dockerfile"}),
                variants=["datalayer"],
                bases=BASES,
            )
        assert raised.value.code.code == "DL_ENV_CAPABILITY_UNSUPPORTED"


# -- The runners --------------------------------------------------------------


class TestTheLocalRunner:
    def test_it_refuses_apt_rather_than_pin_the_wrong_distribution(self) -> None:
        request = ResolveRequest(
            python_version="3.13",
            requirements=("geopandas==1.1.1",),
            constraints=(),
            indexes=("https://pypi.org/simple",),
            apt=("gdal-bin",),
        )
        with pytest.raises(EnvironmentsError) as raised:
            LocalResolveRunner(uv="/usr/bin/true").solve(request)
        assert raised.value.code.code == "DL_ENV_CAPABILITY_UNSUPPORTED"
        assert raised.value.detail["apt"] == ["gdal-bin"]

    def test_it_says_so_when_there_is_no_uv(self) -> None:
        request = ResolveRequest(python_version="3.13", requirements=(), constraints=(), indexes=())
        with pytest.raises(EnvironmentsError) as raised:
            LocalResolveRunner(uv="").solve(request)
        assert raised.value.detail["missing"] == "uv"


class TestTheBuildkitRunner:
    def test_the_solve_is_a_dockerfile_from_the_base_that_exports_the_lock(self) -> None:
        from code_sandboxes.environments.resolve import BuildkitResolveRunner

        runner = BuildkitResolveRunner(
            buildctl="/usr/bin/true",
            apt_snapshot="https://snapshot.ubuntu.com/ubuntu/20260901T000000Z",
        )
        dockerfile = runner.dockerfile(
            ResolveRequest(
                python_version="3.13",
                requirements=("geopandas==1.1.1",),
                constraints=("ipykernel==7.3.0",),
                indexes=("https://pypi.org/simple",),
                apt=("gdal-bin",),
                base_reference="environments/base/python-cpu@sha256:" + "11" * 32,
            )
        )
        assert dockerfile.startswith("FROM environments/base/python-cpu@sha256:")
        assert "uv pip compile" in dockerfile
        assert "--generate-hashes" in dockerfile
        assert "--constraint constraints.txt" in dockerfile
        # A protected pin's own wheel, for what no index has (E1-04, E1-05).
        assert "--find-links /opt/datalayer/wheelhouse" in dockerfile
        assert "snapshot.ubuntu.com" in dockerfile
        assert "apt-get install --simulate" in dockerfile
        # Exported, not left in the image: the lock is the only output.
        assert "FROM scratch" in dockerfile
        assert "COPY --from=solve /solve/lock.txt /lock.txt" in dockerfile

    def test_an_imported_image_bootstraps_uv_from_its_own_wheelhouse(self) -> None:
        """An imported image is not baked with `uv` or the fork's wheel (E3-04):
        both are brought to the solve instead of assumed already there."""
        from code_sandboxes.environments.resolve import BuildkitResolveRunner

        dockerfile = BuildkitResolveRunner(buildctl="/usr/bin/true").dockerfile(
            ResolveRequest(
                python_version="3.13",
                requirements=("ipykernel==7.3.0",),
                constraints=(),
                indexes=(),
                base_reference="docker.io/library/python@sha256:" + "9" * 64,
                bootstrap_uv=True,
            )
        )
        assert 'RUN pip install --no-cache-dir "uv==0.12.11"' in dockerfile
        assert "COPY wheelhouse/ ./wheelhouse/" in dockerfile
        assert "--find-links /solve/wheelhouse" in dockerfile
        assert "/opt/datalayer/wheelhouse" not in dockerfile

    def test_an_approved_base_never_copies_a_wheelhouse_in(self) -> None:
        """`bootstrap_uv` defaults to `False`: an approved base's own solve is
        unchanged by this, since it already has both (E1-05)."""
        from code_sandboxes.environments.resolve import BuildkitResolveRunner

        dockerfile = BuildkitResolveRunner(buildctl="/usr/bin/true").dockerfile(
            ResolveRequest(
                python_version="3.13",
                requirements=(),
                constraints=(),
                indexes=(),
                base_reference="environments/base/python-cpu@sha256:" + "11" * 32,
            )
        )
        assert "wheelhouse/ ./wheelhouse/" not in dockerfile
        assert "uv==0.12.11" not in dockerfile

    def test_the_solve_copies_the_wheelhouse_into_the_build_context_when_asked(
        self, monkeypatch
    ) -> None:
        """The context `--local context=<dir>` names holds the wheelhouse
        `COPY wheelhouse/` reads, written before `buildctl` is ever run — so
        it is only there to see while that subprocess call is live."""
        import subprocess as subprocess_module
        from pathlib import Path

        from code_sandboxes.environments import resolve as resolve_module
        from code_sandboxes.environments.resolve import (
            WHEELHOUSE_PATH,
            BuildkitResolveRunner,
        )

        expected = {wheel.name for wheel in WHEELHOUSE_PATH.glob("*.whl")}
        assert expected, "the package's own wheelhouse must not be empty"
        seen: dict[str, set[str]] = {}

        def fake_run(command, **kwargs):
            context = Path(command[command.index("--local") + 1].removeprefix("context="))
            seen["wheelhouse"] = {path.name for path in (context / "wheelhouse").glob("*.whl")}
            return subprocess_module.CompletedProcess(command, 1, "", "boom")

        monkeypatch.setattr(resolve_module.subprocess, "run", fake_run)
        runner = BuildkitResolveRunner(buildctl="/usr/bin/true")
        with pytest.raises(EnvironmentsError):
            runner.solve(
                ResolveRequest(
                    python_version="3.13",
                    requirements=("ipykernel==7.3.0",),
                    constraints=(),
                    indexes=(),
                    base_reference="docker.io/library/python@sha256:" + "9" * 64,
                    bootstrap_uv=True,
                )
            )
        assert seen["wheelhouse"] == expected

    def test_it_refuses_a_base_that_is_not_pinned_by_digest(self) -> None:
        from code_sandboxes.environments.resolve import BuildkitResolveRunner

        with pytest.raises(EnvironmentsError) as raised:
            BuildkitResolveRunner(buildctl="/usr/bin/true").solve(
                ResolveRequest(
                    python_version="3.13",
                    requirements=(),
                    constraints=(),
                    indexes=(),
                    base_reference="datalayer/python-cpu:2026.09",
                )
            )
        assert raised.value.code.code == "DL_ENV_SPEC_INVALID"

    def test_it_names_the_build_pool_when_there_is_no_buildctl(self) -> None:
        from code_sandboxes.environments.resolve import BuildkitResolveRunner

        with pytest.raises(EnvironmentsError) as raised:
            BuildkitResolveRunner(buildctl="").solve(
                ResolveRequest(python_version="3.13", requirements=(), constraints=(), indexes=())
            )
        assert raised.value.detail == {
            "missing": "buildctl",
            "runner": "buildkit",
            "item": "E1-06",
        }


@pytest.mark.live
def test_a_real_uv_resolve_locks_the_section_4_1_example() -> None:
    """The whole thing, against PyPI: needs the network, so it is a live test.

    The p95 E1-04 asks for is measured on r1, through the BuildKit solve, not
    here; this is the check that the argv, the constraints and the lock's
    assembly hold together against the real resolver.
    """
    answer = resolve_environment(
        spec=a_spec(), variants=["datalayer"], runner=LocalResolveRunner(), bases=BASES
    )
    locked = locked_versions(answer["content"])
    assert locked["geopandas"] == "1.1.1"
    assert locked["rasterio"] == "1.4.3"
    assert answer["package_count"] > 10
    assert "--hash=sha256:" in answer["content"]
