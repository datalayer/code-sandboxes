# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Every variant answers whether it can build a spec (PLAN_ENV.md §6, E2-06).

A build that cannot work is said so **before anything is queued**. The cost of
not doing that is concrete: a GPU spec sent to E2B queues a build, waits for a
worker, resolves, uploads, and fails ten minutes later with whatever E2B says
about a template it could not compile — where the true answer, "E2B has no
GPU, use modal or daytona", was available the moment the spec was read.

Two properties are load-bearing here and are tested rather than assumed:

- **The answers are each provider's real constraints** (section 6's table), so
  the message names the provider and what to do about it.
- **No provider SDK is needed to answer.** Runtimes serves
  `POST /environment-versions/{uid}/validate` for every variant and installs
  no provider extra (D-7), so the last test in this file blocks every SDK
  import and validates all four variants anyway. An SDK imported at module
  scope would turn that route into a 500 on the deployment that actually runs
  it.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from typing import Any

import pytest
import yaml

from code_sandboxes.environments.builders import CapabilityReport, get_builder
from code_sandboxes.environments.errors import EnvironmentsError
from code_sandboxes.environments.spec import VARIANTS, Environment, parse_environment

MANAGED = ("e2b", "daytona", "modal")

ENVIRONMENT = (
    "metadata: {name: geo}\n"
    "spec:\n"
    '  language: {version: "3.13"}\n'
    '  base: {ref: datalayer/python-cpu, channel: "2026.09"}\n'
)


def environment(**spec: Any) -> Environment:
    data = yaml.safe_load(ENVIRONMENT)
    data["spec"].update(spec)
    return parse_environment(data)


#: A `dependencyFile` conda source: an `environment.yml` a managed variant
#: builds with `micromamba` (E3-02).
CONDA_ENVIRONMENT_YML = (
    "name: geo\nchannels: [conda-forge]\ndependencies:\n  - python=3.13\n  - gdal\n"
)


def a_conda_environment(**spec: Any) -> Environment:
    return environment(
        build={
            "source": "dependencyFile",
            "dependencyFile": {
                "sourceFormat": "conda",
                "content": CONDA_ENVIRONMENT_YML,
            },
        },
        **spec,
    )


def messages(report: CapabilityReport) -> str:
    return " | ".join(finding.message for finding in report.findings)


def fields(report: CapabilityReport) -> list[str | None]:
    return [finding.field for finding in report.findings]


# -- what each one can do -------------------------------------------------------


class TestTheCapabilitySets:
    def test_every_variant_has_one_that_describes_itself(self) -> None:
        for variant in VARIANTS:
            capabilities = get_builder(variant).capabilities()
            assert capabilities.variant == variant
            assert capabilities.build_sources, variant
            assert capabilities.package_managers, variant

    def test_e2b_has_no_gpu_and_the_other_two_do(self) -> None:
        """Firecracker microVMs have no GPU passthrough; Modal and Daytona run
        one on their own hardware (D-20, E2-17)."""
        assert get_builder("e2b").capabilities().supports_gpu is False
        assert get_builder("modal").capabilities().supports_gpu is True
        assert get_builder("daytona").capabilities().supports_gpu is True

    def test_each_variant_forbids_what_its_own_builder_will_not_honour(self) -> None:
        """Modal implements its own Dockerfile builder, and E2B parses a
        Dockerfile into Template SDK calls; Daytona hands the text to a real
        Docker builder, which implements all of them.

        E2B's list is read from its SDK (E3-03, 2026-09-17):
        `e2b.template.dockerfile_parser` branches on FROM, RUN, COPY, ADD,
        WORKDIR, USER, ENV, ARG, CMD and ENTRYPOINT, and for anything else
        **prints `Unsupported instruction` and carries on** — so a template
        built from a Dockerfile naming one of these comes back without it and
        reports success. That is the case a capability report exists for.
        """
        assert set(get_builder("modal").capabilities().forbidden_instructions) == {
            "ONBUILD",
            "STOPSIGNAL",
            "VOLUME",
        }
        assert set(get_builder("e2b").capabilities().forbidden_instructions) == {
            "VOLUME",
            "EXPOSE",
            "HEALTHCHECK",
            "SHELL",
            "ONBUILD",
            "STOPSIGNAL",
            "LABEL",
            "MAINTAINER",
        }
        assert get_builder("daytona").capabilities().forbidden_instructions == ()

    def test_each_one_bounds_how_long_a_build_may_take(self) -> None:
        """A build with no bound is a worker held for as long as a provider
        feels like (§13, §14)."""
        for variant in MANAGED:
            seconds = get_builder(variant).capabilities().max_build_seconds
            assert seconds and 0 < seconds <= 60 * 60, variant

    def test_this_phase_builds_a_package_list_a_conda_file_and_a_dockerfile(self) -> None:
        """All three take a Dockerfile (E3-03), each through its own door:
        E2B parses one into Template SDK calls, Daytona hands the text to a
        real Docker builder, Modal builds it with its own frontend."""
        for variant in MANAGED:
            sources = get_builder(variant).capabilities().build_sources
            assert sources == ("packages", "dependencyFile", "dockerfile"), variant


# -- what each one refuses ------------------------------------------------------


class TestWhatIsSaidBeforeAnythingIsQueued:
    def test_a_plain_spec_is_buildable_everywhere(self) -> None:
        for variant in VARIANTS:
            report = get_builder(variant).validate(environment())
            assert report.supported is True, f"{variant}: {messages(report)}"

    def test_a_gpu_spec_with_e2b_is_refused_naming_e2b(self) -> None:
        report = get_builder("e2b").validate(
            environment(resources={"sizeClass": "gpu-small", "accelerator": {"type": "A10G"}})
        )
        assert report.supported is False
        assert "E2B has no GPU" in messages(report)
        # And says where a GPU environment can go instead.
        assert "modal" in messages(report) and "daytona" in messages(report)
        assert "spec.resources.sizeClass" in fields(report)

    def test_an_accelerator_alone_is_the_same_ask(self) -> None:
        """The spec's own validation couples the class and the accelerator; a
        `validate` of a draft can be reached before that."""
        report = get_builder("e2b").validate(
            environment(resources={"accelerator": {"type": "A10G", "count": 2}})
        )
        assert report.supported is False
        assert "spec.resources.accelerator" in fields(report)

    def test_a_gpu_spec_is_buildable_on_modal_and_daytona(self) -> None:
        spec = {"sizeClass": "gpu-large", "accelerator": {"type": "A100", "cuda": "12.4"}}
        for variant in ("modal", "daytona"):
            report = get_builder(variant).validate(environment(resources=spec))
            assert report.supported is True, f"{variant}: {messages(report)}"

    def test_a_volume_is_refused_by_modal_in_section_sixs_words(self) -> None:
        """Modal's builder would accept the line and do nothing, which is the
        worst of the three possible answers."""
        report = get_builder("modal").validate(
            environment(commands={"postInstall": ["VOLUME /data"]})
        )
        assert report.supported is False
        assert (
            "`VOLUME` is not supported by the Modal builder. Remove it, or drop `modal` from "
            "the optional variants" in messages(report)
        )
        assert fields(report) == ["spec.commands.postInstall[0]"]

    @pytest.mark.parametrize("instruction", ["ONBUILD", "STOPSIGNAL", "VOLUME"])
    def test_each_unimplemented_instruction_is_named(self, instruction: str) -> None:
        report = get_builder("modal").validate(
            environment(commands={"postInstall": ["echo one", f"{instruction.lower()} thing"]})
        )
        assert f"`{instruction}` is not supported by the Modal builder" in messages(report)
        assert fields(report) == ["spec.commands.postInstall[1]"]

    def test_the_same_commands_build_on_the_other_variants(self) -> None:
        """Only Modal's own builder is missing them."""
        for variant in ("e2b", "daytona", "datalayer"):
            report = get_builder(variant).validate(
                environment(commands={"postInstall": ["VOLUME /data"]})
            )
            assert report.supported is True, f"{variant}: {messages(report)}"

    def test_a_moving_tag_is_refused_by_daytona(self) -> None:
        """Daytona refuses `latest`, `lts` and `stable` as a snapshot's source
        image: each moves, and a snapshot is built once."""
        for tag in ("latest", "LTS", "stable"):
            report = get_builder("daytona").validate(
                environment(base={"ref": "datalayer/python-cpu", "channel": tag})
            )
            assert report.supported is False, tag
            assert "because it moves" in messages(report)
            assert "spec.base.channel" in fields(report)

    def test_a_dated_channel_is_what_it_takes(self) -> None:
        report = get_builder("daytona").validate(
            environment(base={"ref": "datalayer/python-cpu", "channel": "2026.09"})
        )
        assert report.supported is True, messages(report)

    def test_several_regions_are_several_daytona_artifacts_and_it_says_so(self) -> None:
        report = get_builder("daytona").validate(
            environment(compatibility={"regions": ["us", "eu"]})
        )
        assert "2 regions are 2 artifacts" in messages(report)

    def test_one_region_is_one_artifact_and_says_nothing(self) -> None:
        report = get_builder("daytona").validate(environment(compatibility={"regions": ["us"]}))
        assert report.supported is True, messages(report)

    def test_a_home_of_its_own_is_refused_by_e2b(self) -> None:
        """E2B ignores the image's user and adds a sudo user of its own, so an
        `env.HOME` would not be what a sandbox sees."""
        report = get_builder("e2b").validate(environment(env={"HOME": "/opt/me"}))
        assert report.supported is False
        assert "E2B sets the sandbox's HOME itself" in messages(report)
        assert "spec.env.HOME" in fields(report)

    def test_other_variables_are_left_alone(self) -> None:
        report = get_builder("e2b").validate(environment(env={"MY_TOKEN_NAME": "x"}))
        assert report.supported is True, messages(report)

    def test_a_source_this_phase_does_not_build_is_refused_by_name(self) -> None:
        for variant in MANAGED:
            report = get_builder(variant).validate(environment(build={"source": "image"}))
            assert report.supported is False, variant
            assert "`image` is not built for" in messages(report)
            assert "it builds packages, dependencyFile, dockerfile" in messages(report)

    def test_an_instruction_a_variant_would_drop_is_refused_with_its_line(self) -> None:
        """The case this exists for: E2B's parser prints `Unsupported
        instruction` and carries on, so without this the template comes back
        missing what the Dockerfile asked for and the build reports success."""
        dockerfile = "FROM datalayer/python-cpu:2026.09\nRUN true\nVOLUME /data\n"
        report = get_builder("e2b").validate(
            environment(build={"source": "dockerfile", "dockerfile": {"content": dockerfile}})
        )
        assert report.supported is False
        assert "line 3: E2B does not implement `VOLUME`" in messages(report)

    def test_a_dockerfile_a_variant_can_honour_is_accepted(self) -> None:
        """Daytona builds the text as it is, so Docker's own grammar is the limit."""
        dockerfile = "FROM datalayer/python-cpu:2026.09\nVOLUME /data\nEXPOSE 8888\n"
        report = get_builder("daytona").validate(
            environment(build={"source": "dockerfile", "dockerfile": {"content": dockerfile}})
        )
        assert report.supported is True, messages(report)

    def test_a_conda_dependency_file_is_buildable_on_every_managed_variant(self) -> None:
        for variant in MANAGED:
            report = get_builder(variant).validate(a_conda_environment())
            assert report.supported is True, f"{variant}: {messages(report)}"

    def test_a_pyproject_dependency_file_is_not_built_for_a_managed_variant_yet(self) -> None:
        report = get_builder("e2b").validate(
            environment(
                build={
                    "source": "dependencyFile",
                    "dependencyFile": {
                        "sourceFormat": "pyproject",
                        "content": "[project]\nname='x'\nversion='0'\n",
                        "lockContent": "# lock\n",
                    },
                }
            )
        )
        assert report.supported is False
        assert "a `pyproject` dependency file is not built for E2B yet" in messages(report)
        assert "spec.build.dependencyFile.sourceFormat" in fields(report)

    @pytest.mark.parametrize(
        "dependency_file",
        [
            {"sourceFormat": "requirements", "content": "geopandas==1.1.1\n"},
            {
                "sourceFormat": "pyproject",
                "content": "[project]\nname='x'\nversion='0'\n",
                "lockContent": "version = 1\n",
            },
        ],
        ids=["requirements", "pyproject"],
    )
    def test_daytona_builds_a_pip_dependency_file(self, dependency_file: dict) -> None:
        """Both resolve to the pip lock a `packages` list does (E3-01), and
        Daytona's build installs that lock whatever wrote it. Refusing them
        kept E3-08's first two examples and half of E3-09 off Daytona."""
        report = get_builder("daytona").validate(
            environment(build={"source": "dependencyFile", "dependencyFile": dependency_file})
        )
        assert report.supported is True, messages(report)

    def test_e2b_and_daytona_refuse_a_build_secret_e0_04_found_no_mechanism_for(self) -> None:
        """E0-04's spike found only a registry login for the private base on
        either provider, never a per-step arbitrary named secret: a spec
        naming one is refused before anything is queued, rather than
        silently dropped or baked into the image (E3-05)."""
        spec = {"buildSecrets": [{"id": "dlsec_01J9BUILDSECRET0000000000", "name": "TOKEN"}]}
        for variant in ("e2b", "daytona"):
            report = get_builder(variant).validate(environment(**spec))
            assert report.supported is False, variant
            assert "no per-step secret mechanism" in messages(report)
            assert "datalayer" in messages(report) and "modal" in messages(report)
            assert "spec.buildSecrets" in fields(report)

    def test_datalayer_accepts_a_build_secret(self) -> None:
        spec = {"buildSecrets": [{"id": "dlsec_01J9BUILDSECRET0000000000", "name": "TOKEN"}]}
        report = get_builder("datalayer").validate(environment(**spec))
        assert report.supported is True, messages(report)

    def test_modal_accepts_an_env_build_secret_and_refuses_a_file_one(self) -> None:
        """Modal has a per-step `secrets=` mechanism, unlike E2B and Daytona,
        and attaches a secret to the steps that name it (E3-05). It passes
        environment variables only, so a `mountAs: file` secret, which would
        have to be written into a layer, is refused."""
        secret = {"id": "dlsec_01J9BUILDSECRET0000000000", "name": "TOKEN"}
        report = get_builder("modal").validate(environment(buildSecrets=[secret]))
        assert "spec.buildSecrets" not in " ".join(fields(report)), messages(report)
        report = get_builder("modal").validate(
            environment(buildSecrets=[{**secret, "mountAs": "file"}])
        )
        assert report.supported is False, messages(report)
        assert "spec.buildSecrets[0].mountAs" in fields(report)

    def test_every_finding_carries_the_capability_code(self) -> None:
        report = get_builder("e2b").validate(
            environment(
                build={"source": "image"},
                resources={"sizeClass": "gpu-small", "accelerator": {"type": "A10G"}},
            )
        )
        assert len(report.findings) == 2
        assert {finding.code for finding in report.findings} == {"DL_ENV_CAPABILITY_UNSUPPORTED"}


# -- what still needs the provider ----------------------------------------------


class TestTheHalfThatIsNotBuiltYet:
    """E2B (E2-03), Daytona (E2-04) and Modal (E2-05) — every managed
    variant — are all built now, each covered by its own
    test_environment_{e2b,daytona,modal}_builder.py, so there is no longer a
    variant to parametrize "refuses because it is not built at all yet"
    over. What is left standing is what each one's own item did not build:
    `smoke_test`, `resolve` and `delete`, still refusing via the inherited
    `ManagedBuilder` methods on all three."""

    def test_modal_still_refuses_what_e2_05_did_not_build(self) -> None:
        """`build`/`inspect`/`exists` are E2-05's, `delete` is too — it
        collects the intermediate layers a build recorded (E2-05, E2-09) —
        and so is `smoke_test`, since without it no Modal build could reach
        `succeeded`. `resolve` is still not built — see
        test_environment_modal_builder.py for what is."""
        builder = get_builder("modal")
        calls = {
            "resolve": lambda: builder.resolve("geo@1"),
        }
        for operation, call in calls.items():
            with pytest.raises(EnvironmentsError) as raised:
                call()
            assert raised.value.detail["operation"], f"modal.{operation}"

    def test_e2b_still_refuses_what_e2_03_did_not_build(self) -> None:
        """`build`/`inspect`/`exists` are E2-03's; `smoke_test`, `resolve` and
        `delete` are not — see adapters/e2b.py's own module docstring for
        `smoke_test`, and test_environment_e2b_builder.py for what is built."""
        builder = get_builder("e2b")
        calls = {
            "smoke_test": lambda: builder.smoke_test(None),  # type: ignore[arg-type]
            "resolve": lambda: builder.resolve("geo@1"),
            "delete": lambda: builder.delete(None),  # type: ignore[arg-type]
        }
        for operation, call in calls.items():
            with pytest.raises(EnvironmentsError) as raised:
                call()
            assert raised.value.detail["operation"], f"e2b.{operation}"

    def test_daytona_still_refuses_what_e2_04_did_not_build(self) -> None:
        """`build`, `inspect`, `exists` and now `smoke_test` are E2-04's —
        the last of them because this box's own `Done when` asks for "a
        sandbox launched from its id passes the core tier", and until it was
        built no Daytona build could reach `succeeded`. `delete` is E2-18's.
        `resolve` is still not built — see test_environment_daytona_builder.py."""
        builder = get_builder("daytona")
        calls = {
            "resolve": lambda: builder.resolve("geo@1"),
        }
        for operation, call in calls.items():
            with pytest.raises(EnvironmentsError) as raised:
                call()
            assert raised.value.detail["operation"], f"daytona.{operation}"


# -- no provider SDK ------------------------------------------------------------


class _NoProviderSdk:
    """Every provider SDK missing, whatever is installed in this interpreter.

    Runtimes installs no provider extra (D-7) and still answers `validate` for
    every variant, so that deployment is what this reproduces.
    """

    BLOCKED = ("e2b", "e2b_code_interpreter", "daytona", "daytona_sdk", "modal", "modal_proto")

    def __init__(self) -> None:
        self.asked: list[str] = []

    def __enter__(self) -> _NoProviderSdk:
        self._real_import = builtins.__import__
        self._removed = {
            name: module
            for name, module in list(sys.modules.items())
            if name.split(".")[0] in self.BLOCKED
            or name.startswith("code_sandboxes.environments.adapters.")
        }
        for name in self._removed:
            del sys.modules[name]

        def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.split(".")[0] in self.BLOCKED:
                self.asked.append(name)
                raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}", name=name)
            return self._real_import(name, *args, **kwargs)

        builtins.__import__ = blocked
        return self

    def __exit__(self, *_exception: Any) -> None:
        builtins.__import__ = self._real_import
        for name in list(sys.modules):
            if name.startswith("code_sandboxes.environments.adapters."):
                del sys.modules[name]
        sys.modules.update(self._removed)
        importlib.invalidate_caches()


def test_every_variant_validates_with_no_provider_sdk_installed() -> None:
    """The one this item turns on (E2-06).

    Nothing in an adapter's import or its capability half may reach for an
    SDK: Runtimes has none, and a 500 on `validate` would make the only
    pre-queue answer unavailable exactly where it is asked for.
    """
    with _NoProviderSdk() as without:
        for variant in VARIANTS:
            builder = get_builder(variant)
            assert builder.capabilities().variant == variant
            report = builder.validate(environment())
            assert report.supported is True, f"{variant}: {messages(report)}"
            # A refusal each variant still makes, and makes without an SDK.
            # `dockerfile` stopped being one when E3-03 gave all three a door
            # into it, so a managed variant is asked about `image` instead.
            refused = get_builder(variant).validate(
                environment(build={"source": "image"})
                if variant != "datalayer"
                else environment(
                    resources={"sizeClass": "gpu-small", "accelerator": {"type": "A10G"}}
                )
            )
            assert refused.supported is False, variant
        assert without.asked == [], f"an adapter reached for {without.asked}"
