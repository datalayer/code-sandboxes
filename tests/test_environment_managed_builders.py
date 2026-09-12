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

    def test_only_modal_forbids_instructions_its_builder_never_implemented(self) -> None:
        """Modal implements its own Dockerfile builder; the others hand a
        Dockerfile to BuildKit, which implements all of them."""
        assert set(get_builder("modal").capabilities().forbidden_instructions) == {
            "ONBUILD",
            "STOPSIGNAL",
            "VOLUME",
        }
        assert get_builder("e2b").capabilities().forbidden_instructions == ()
        assert get_builder("daytona").capabilities().forbidden_instructions == ()

    def test_each_one_bounds_how_long_a_build_may_take(self) -> None:
        """A build with no bound is a worker held for as long as a provider
        feels like (§13, §14)."""
        for variant in MANAGED:
            seconds = get_builder(variant).capabilities().max_build_seconds
            assert seconds and 0 < seconds <= 60 * 60, variant

    def test_this_phase_builds_a_package_list_and_nothing_else(self) -> None:
        for variant in MANAGED:
            assert get_builder(variant).capabilities().build_sources == ("packages",)


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
            report = get_builder(variant).validate(environment(build={"source": "dockerfile"}))
            assert report.supported is False, variant
            assert "`dockerfile` is not built for" in messages(report)
            assert "it builds packages" in messages(report)

    def test_conda_is_not_resolved_for_a_managed_variant_yet(self) -> None:
        report = get_builder("e2b").validate(
            environment(packages={"python": {"manager": "conda", "dependencies": ["numpy"]}})
        )
        assert report.supported is False
        assert "`conda` is not resolved for E2B yet" in messages(report)

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
    @pytest.mark.parametrize(
        "variant,item", [("e2b", "E2-03"), ("daytona", "E2-04"), ("modal", "E2-05")]
    )
    def test_a_build_refuses_by_naming_what_is_missing(self, variant: str, item: str) -> None:
        """A caller is never told "no builder" when the real answer is "not
        this item yet" — and never when the real answer is "not this spec"."""
        builder = get_builder(variant)
        with pytest.raises(EnvironmentsError) as raised:
            builder.build(None)  # type: ignore[arg-type]
        assert raised.value.code.code == "DL_ENV_CAPABILITY_UNSUPPORTED"
        assert raised.value.detail["missing"] == item
        assert raised.value.detail["variant"] == variant
        assert "answer whether a spec is buildable" in str(raised.value)

    @pytest.mark.parametrize("variant", MANAGED)
    def test_every_operation_that_touches_the_provider_refuses(self, variant: str) -> None:
        builder = get_builder(variant)
        calls = {
            "build": lambda: builder.build(None),  # type: ignore[arg-type]
            "inspect": lambda: builder.inspect(None),  # type: ignore[arg-type]
            "smoke_test": lambda: builder.smoke_test(None),  # type: ignore[arg-type]
            "resolve": lambda: builder.resolve("geo@1"),
            "exists": lambda: builder.exists(None),  # type: ignore[arg-type]
            "delete": lambda: builder.delete(None),  # type: ignore[arg-type]
        }
        for operation, call in calls.items():
            with pytest.raises(EnvironmentsError) as raised:
                call()
            assert raised.value.detail["operation"], f"{variant}.{operation}"


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
            refused = get_builder(variant).validate(
                environment(build={"source": "dockerfile"})
                if variant != "datalayer"
                else environment(
                    resources={"sizeClass": "gpu-small", "accelerator": {"type": "A10G"}}
                )
            )
            assert refused.supported is False, variant
        assert without.asked == [], f"an adapter reached for {without.asked}"
