# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Conda resolution: an environment.yml to one explicit lock (E3-02).

Like the pip resolver's suite, micromamba's refusals below are **recorded** —
each is what ``micromamba`` actually writes for that input — so the parser is
tested against the solver's own words rather than a paraphrase of them.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

import pytest

from code_sandboxes.environments.bases import ApprovedBase
from code_sandboxes.environments.errors import EnvironmentsError
from code_sandboxes.environments.resolve import protected_pins
from code_sandboxes.environments.resolve_conda import (
    CONDA_LOCK_FORMAT,
    BuildkitCondaResolveRunner,
    CondaResolveOutcome,
    CondaResolveRequest,
    MicromambaResolveRunner,
    conda_lock_document,
    explicit_lock_packages,
    merge_conda_pip,
    parse_conda_environment,
    parse_conda_failure,
    pip_requirements_from_env_yaml,
    rendered_environment,
    resolve_conda_environment,
)
from code_sandboxes.environments.spec import validate_environment

# -- What micromamba wrote ----------------------------------------------------

#: An explicit lock, as ``micromamba env export --explicit`` writes one.
EXPLICIT_LOCK = """\
# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: linux-64
@EXPLICIT
https://conda.anaconda.org/conda-forge/linux-64/python-3.13.0.conda#{}
https://conda.anaconda.org/conda-forge/linux-64/gdal-3.9.2.conda#{}
""".format("aa" * 32, "bb" * 32)

#: A package no channel serves.
CONDA_MISSING = """\
critical libmamba Could not solve for environment specs
The following package could not be found:
  - nothing provides no-such-conda-pkg-xyzzy needed by requested
"""

#: An unsatisfiable set.
CONDA_CONFLICT = """\
critical libmamba Could not solve for environment specs
The following packages are incompatible
encountered problems while solving:
  - package gdal-3.9.2 requires libgdal 3.9.*, but none of the providers can be installed
"""

#: Not a resolution failure at all — the channel is unreachable.
CONDA_UNREACHABLE = """\
critical libmamba Download error (6) Could not resolve host: conda.anaconda.org
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

A_YAML = """\
name: geospatial
channels:
  - conda-forge
dependencies:
  - gdal=3.9
  - pip:
    - shapely==2.0.6
"""


def a_conda_spec(content: str = A_YAML, **spec: object) -> dict[str, object]:
    """A section 4.1 example whose build source is a conda environment.yml."""
    return {
        "apiVersion": "environments.datalayer.io/v1alpha1",
        "kind": "Environment",
        "metadata": {"name": "geospatial", "title": "Geospatial"},
        "spec": {
            "language": {"name": "python", "version": "3.13"},
            "base": {"ref": "datalayer/python-cpu", "channel": "2026.09"},
            "build": {
                "source": "dependencyFile",
                "dependencyFile": {"sourceFormat": "conda", "content": content},
            },
            "resources": {"sizeClass": "medium"},
            "compatibility": {"variants": {"required": ["datalayer"], "optional": ["modal"]}},
            **spec,
        },
    }


class RecordedRunner:
    """A conda solve that answers what it was given, remembering the request."""

    name = "recorded-conda"

    def __init__(self, outcome: CondaResolveOutcome | Exception) -> None:
        self._outcome = outcome
        self.request: CondaResolveRequest | None = None

    def solve(self, request: CondaResolveRequest, log=None) -> CondaResolveOutcome:
        self.request = request
        if log is not None:
            log("solving conda")
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


# -- Reading the environment.yml ---------------------------------------------


class TestParsingTheEnvironmentYml:
    def test_it_separates_the_conda_and_pip_layers(self) -> None:
        env = parse_conda_environment(A_YAML)
        assert env.channels == ("conda-forge",)
        assert env.conda_dependencies == ("gdal=3.9",)
        assert env.pip_dependencies == ("shapely==2.0.6",)

    def test_no_pip_section_is_an_empty_pip_layer(self) -> None:
        env = parse_conda_environment("dependencies:\n  - gdal=3.9\n")
        assert env.conda_dependencies == ("gdal=3.9",)
        assert env.pip_dependencies == ()

    def test_malformed_yaml_is_refused_with_its_field(self) -> None:
        with pytest.raises(EnvironmentsError) as caught:
            parse_conda_environment("dependencies: [\n")
        assert caught.value.code.code == "DL_ENV_SPEC_INVALID"
        assert caught.value.detail["field"] == "spec.build.dependencyFile.content"

    def test_a_document_that_is_not_a_mapping_is_refused(self) -> None:
        with pytest.raises(EnvironmentsError):
            parse_conda_environment("- just\n- a\n- list\n")

    def test_missing_dependencies_is_refused(self) -> None:
        with pytest.raises(EnvironmentsError) as caught:
            parse_conda_environment("name: env\nchannels: [conda-forge]\n")
        assert "dependencies" in caught.value.message

    def test_two_pip_sections_are_refused(self) -> None:
        text = "dependencies:\n  - pip:\n    - a\n  - pip:\n    - b\n"
        with pytest.raises(EnvironmentsError) as caught:
            parse_conda_environment(text)
        assert "more than one" in caught.value.message

    def test_a_nested_list_entry_is_refused(self) -> None:
        with pytest.raises(EnvironmentsError):
            parse_conda_environment("dependencies:\n  - [nested]\n")

    def test_a_non_string_pip_entry_is_refused(self) -> None:
        with pytest.raises(EnvironmentsError):
            parse_conda_environment("dependencies:\n  - pip:\n    - 3\n")

    def test_channels_that_are_not_a_list_are_refused(self) -> None:
        with pytest.raises(EnvironmentsError):
            parse_conda_environment("channels: conda-forge\ndependencies:\n  - gdal\n")

    def test_a_channel_url_carrying_a_credential_is_refused(self) -> None:
        # The same refusal a credential-bearing pip index gets: the token
        # belongs in the build's secrets, never verbatim in the spec.
        text = (
            "channels:\n"
            "  - https://user:tok@conda.example.com/private\n"
            "dependencies:\n  - gdal\n"
        )
        with pytest.raises(EnvironmentsError) as caught:
            parse_conda_environment(text)
        assert caught.value.code.code == "DL_ENV_SPEC_INVALID"
        assert "credential" in caught.value.message
        assert caught.value.detail["field"].endswith("channels[0]")

    def test_a_plain_channel_url_without_a_credential_is_kept(self) -> None:
        env = parse_conda_environment(
            "channels:\n  - https://conda.anaconda.org/conda-forge\n"
            "dependencies:\n  - gdal\n"
        )
        assert env.channels == ("https://conda.anaconda.org/conda-forge",)


class TestReadingThePipExport:
    def test_it_reads_the_pip_section_in_order(self) -> None:
        export = (
            "name: solved\n"
            "channels:\n  - conda-forge\n"
            "dependencies:\n"
            "  - python=3.13\n"
            "  - gdal=3.9.2\n"
            "  - pip:\n"
            "    - shapely==2.0.6\n"
            "    - ipykernel==7.3.0\n"
        )
        assert pip_requirements_from_env_yaml(export) == (
            "shapely==2.0.6",
            "ipykernel==7.3.0",
        )

    def test_an_export_without_a_pip_section_is_an_empty_layer(self) -> None:
        export = "dependencies:\n  - python=3.13\n  - gdal=3.9.2\n"
        assert pip_requirements_from_env_yaml(export) == ()

    def test_a_malformed_export_is_an_empty_layer_never_a_raise(self) -> None:
        assert pip_requirements_from_env_yaml("dependencies: [\n") == ()
        assert pip_requirements_from_env_yaml("- just\n- a\n- list\n") == ()


# -- Datalayer's pins over the pip layer -------------------------------------


class TestMergingThePipLayer:
    def test_the_protected_pins_are_forced_into_the_pip_layer(self) -> None:
        env = parse_conda_environment(A_YAML)
        merged = merge_conda_pip(env)
        names = {req.split("==")[0] for req in merged.requirements}
        assert {"shapely", "ipykernel", "jupyter-server"} <= names

    def test_a_pip_requirement_contradicting_a_pin_is_refused(self) -> None:
        env = parse_conda_environment(
            "dependencies:\n  - pip:\n    - ipykernel==6.0.0\n"
        )
        with pytest.raises(EnvironmentsError) as caught:
            merge_conda_pip(env)
        assert caught.value.code.code == "DL_ENV_PROTECTED_PACKAGE"

    def test_the_interpreter_is_pinned_and_never_doubled(self) -> None:
        env = parse_conda_environment(
            "dependencies:\n  - python=3.11\n  - gdal=3.9\n"
        )
        merged = merge_conda_pip(env)
        rendered = rendered_environment(env, merged, python_version="3.13")
        assert rendered.count("python=3.13") == 1
        assert "python=3.11" not in rendered

    def test_the_rendered_pip_layer_carries_the_pins(self) -> None:
        env = parse_conda_environment(A_YAML)
        merged = merge_conda_pip(env)
        rendered = rendered_environment(env, merged, python_version="3.13")
        assert "ipykernel==7.3.0" in rendered
        assert "shapely==2.0.6" in rendered


# -- Reading micromamba's refusals -------------------------------------------


class TestReadingRefusals:
    def test_a_missing_package_is_package_not_found(self) -> None:
        error = parse_conda_failure(CONDA_MISSING)
        assert error.code.code == "DL_ENV_PACKAGE_NOT_FOUND"
        assert "no-such-conda-pkg-xyzzy" in error.message

    def test_an_unsatisfiable_set_is_resolve_conflict(self) -> None:
        error = parse_conda_failure(CONDA_CONFLICT)
        assert error.code.code == "DL_ENV_RESOLVE_CONFLICT"

    def test_an_unreachable_channel_is_a_provider_error(self) -> None:
        error = parse_conda_failure(CONDA_UNREACHABLE)
        assert error.code.code == "DL_ENV_PROVIDER_ERROR"
        assert error.code.retryable is True


# -- The lock -----------------------------------------------------------------


class TestTheLock:
    def test_it_counts_only_the_package_urls(self) -> None:
        assert explicit_lock_packages(EXPLICIT_LOCK) == [
            "https://conda.anaconda.org/conda-forge/linux-64/python-3.13.0.conda#" + "aa" * 32,
            "https://conda.anaconda.org/conda-forge/linux-64/gdal-3.9.2.conda#" + "bb" * 32,
        ]

    def test_the_document_records_the_pins_and_is_deterministic(self) -> None:
        env = parse_conda_environment(A_YAML)
        merged = merge_conda_pip(env)
        at = datetime(2026, 9, 14, tzinfo=timezone.utc)
        document = conda_lock_document(
            CondaResolveOutcome(lock_text=EXPLICIT_LOCK),
            python_version="3.13",
            base_reference="registry/base@sha256:" + "11" * 32,
            merged=merged,
            resolved_at=at,
        )
        assert document["format"] == CONDA_LOCK_FORMAT
        assert document["package_count"] == 2
        # The header carries the complete pip layer: the user's own pip
        # requirement and the protected pin the resolver forced over it.
        assert "# datalayer-pip: shapely==2.0.6" in document["content"]
        assert "# datalayer-pip: ipykernel==7.3.0" in document["content"]
        assert "@EXPLICIT" in document["content"]
        again = conda_lock_document(
            CondaResolveOutcome(lock_text=EXPLICIT_LOCK),
            python_version="3.13",
            base_reference="registry/base@sha256:" + "11" * 32,
            merged=merged,
            resolved_at=at,
        )
        assert document["digest"] == again["digest"]

    def test_an_export_without_the_marker_is_a_provider_error(self) -> None:
        env = parse_conda_environment(A_YAML)
        merged = merge_conda_pip(env)
        with pytest.raises(EnvironmentsError) as caught:
            conda_lock_document(
                CondaResolveOutcome(lock_text="just some lines\nno marker\n"),
                python_version="3.13",
                base_reference="base@sha256:" + "11" * 32,
                merged=merged,
            )
        assert caught.value.code.code == "DL_ENV_PROVIDER_ERROR"


# -- The runners refuse honestly when their tool is absent -------------------


class TestTheRunners:
    def test_the_local_runner_refuses_without_micromamba(self) -> None:
        runner = MicromambaResolveRunner(micromamba="")
        with pytest.raises(EnvironmentsError) as caught:
            runner.solve(CondaResolveRequest(environment_yml=A_YAML, python_version="3.13"))
        assert caught.value.code.code == "DL_ENV_CAPABILITY_UNSUPPORTED"

    def test_an_export_that_times_out_is_a_provider_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The solve finishes, but the `env export` that reads the lock back
        # exceeds the deadline: its timeout is classified the same as the
        # solve's, never left as a raw subprocess.TimeoutExpired.
        runner = MicromambaResolveRunner(micromamba="/usr/local/bin/micromamba", timeout=5.0)
        calls = {"n": 0}

        def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls["n"] += 1
            if calls["n"] == 1:
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            raise subprocess.TimeoutExpired(argv, 5.0)

        monkeypatch.setattr(
            "code_sandboxes.environments.resolve_conda.subprocess.run", fake_run
        )
        with pytest.raises(EnvironmentsError) as caught:
            runner.solve(CondaResolveRequest(environment_yml=A_YAML, python_version="3.13"))
        assert caught.value.code.code == "DL_ENV_PROVIDER_ERROR"

    def test_the_buildkit_runner_refuses_without_buildctl(self) -> None:
        runner = BuildkitCondaResolveRunner(buildctl="")
        with pytest.raises(EnvironmentsError) as caught:
            runner.solve(
                CondaResolveRequest(
                    environment_yml=A_YAML,
                    python_version="3.13",
                    base_reference="base@sha256:" + "11" * 32,
                )
            )
        assert caught.value.code.code == "DL_ENV_CAPABILITY_UNSUPPORTED"

    def test_the_buildkit_runner_refuses_an_unpinned_base(self) -> None:
        runner = BuildkitCondaResolveRunner(buildctl="/usr/bin/buildctl")
        with pytest.raises(EnvironmentsError) as caught:
            runner.solve(
                CondaResolveRequest(
                    environment_yml=A_YAML, python_version="3.13", base_reference="base:latest"
                )
            )
        assert caught.value.code.code == "DL_ENV_SPEC_INVALID"

    def test_the_buildkit_dockerfile_brings_the_wheelhouse_and_solves(self) -> None:
        runner = BuildkitCondaResolveRunner(buildctl="/usr/bin/buildctl")
        dockerfile = runner.dockerfile(
            CondaResolveRequest(
                environment_yml=A_YAML,
                python_version="3.13",
                base_reference="base@sha256:" + "11" * 32,
            )
        )
        assert "micromamba create" in dockerfile
        assert "PIP_FIND_LINKS" in dockerfile
        assert "micromamba env export --explicit" in dockerfile
        # The pinned micromamba is copied in (the base bakes uv, not it), and
        # the pip layer is exported alongside the explicit lock.
        assert "COPY --from=mambaorg/micromamba" in dockerfile
        assert "env export --prefix /solve/prefix > /solve/pip-env.yml" in dockerfile
        assert "COPY --from=solve /solve/pip-env.yml /pip-env.yml" in dockerfile


# -- The whole resolve, through the recorded runner --------------------------


class TestResolvingACondaVersion:
    def test_it_answers_the_lock_and_the_bases(self) -> None:
        runner = RecordedRunner(CondaResolveOutcome(lock_text=EXPLICIT_LOCK))
        document = resolve_conda_environment(
            environment_yml=A_YAML,
            python_version="3.13",
            resolved_bases={"datalayer": "base@sha256:" + "11" * 32},
            runner=runner,
        )
        assert document["format"] == CONDA_LOCK_FORMAT
        assert document["package_count"] == 2
        assert document["resolved_bases"] == {"datalayer": "base@sha256:" + "11" * 32}

    def test_it_sends_the_rendered_environment_with_the_pins(self) -> None:
        runner = RecordedRunner(CondaResolveOutcome(lock_text=EXPLICIT_LOCK))
        resolve_conda_environment(
            environment_yml=A_YAML,
            python_version="3.13",
            resolved_bases={"datalayer": "base@sha256:" + "11" * 32},
            runner=runner,
        )
        assert runner.request is not None
        assert "python=3.13" in runner.request.environment_yml
        assert "ipykernel==7.3.0" in runner.request.environment_yml

    def test_a_missing_package_propagates_as_package_not_found(self) -> None:
        runner = RecordedRunner(parse_conda_failure(CONDA_MISSING))
        with pytest.raises(EnvironmentsError) as caught:
            resolve_conda_environment(
                environment_yml=A_YAML,
                python_version="3.13",
                resolved_bases={"datalayer": "base@sha256:" + "11" * 32},
                runner=runner,
            )
        assert caught.value.code.code == "DL_ENV_PACKAGE_NOT_FOUND"


# -- The spec validates a conda dependency file ------------------------------


class TestValidatingTheSpec:
    def test_a_conda_environment_file_validates(self) -> None:
        # Does not raise: a well-formed conda source is authorable.
        validate_environment(a_conda_spec(), bases=BASES)

    def test_an_empty_environment_file_is_refused(self) -> None:
        with pytest.raises(EnvironmentsError) as caught:
            validate_environment(a_conda_spec(content="   \n"), bases=BASES)
        assert caught.value.code.code == "DL_ENV_SPEC_INVALID"

    def test_a_malformed_environment_file_is_refused(self) -> None:
        with pytest.raises(EnvironmentsError) as caught:
            validate_environment(a_conda_spec(content="dependencies: [\n"), bases=BASES)
        assert caught.value.code.code == "DL_ENV_SPEC_INVALID"

    def test_a_lock_content_is_refused_for_conda(self) -> None:
        spec = a_conda_spec()
        spec["spec"]["build"]["dependencyFile"]["lockContent"] = "irrelevant"  # type: ignore[index]
        with pytest.raises(EnvironmentsError) as caught:
            validate_environment(spec, bases=BASES)
        assert "pyproject" in caught.value.message


class TestTheProtectedPinsAreTheKernelStack:
    def test_merge_uses_the_same_pins_the_pip_resolver_does(self) -> None:
        env = parse_conda_environment(A_YAML)
        merged = merge_conda_pip(env)
        pin_names = {pin.name for pin in protected_pins()}
        forced = {req.split("==")[0].replace("_", "-") for req in merged.requirements}
        assert pin_names <= forced
