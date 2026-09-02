# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

"""Every sandbox speaks the same lifecycle vocabulary (D32).

The same nouns used to have two spellings — `Sandbox.stop()` here,
`DELETE /runtimes/{pod}` there — so every caller wanting both wrote a
translation layer, and no two translation layers agreed. These tests hold the
convergence in place: one set of verbs, and one way of refusing the ones a
provider cannot do.
"""

from __future__ import annotations

import pytest

from code_sandboxes.base import Sandbox
from code_sandboxes.lifecycle import (
    LIFECYCLE_OPERATIONS,
    SandboxLifecycle,
    SandboxOperationNotSupported,
    unsupported,
)


class TestVocabulary:
    def test_the_verbs_are_named_once(self) -> None:
        assert set(LIFECYCLE_OPERATIONS) == {
            "create",
            "start",
            "stop",
            "pause",
            "resume",
            "list",
            "get",
            "update",
            "snapshot",
            "execute",
        }

    def test_each_verb_records_its_rest_spelling(self) -> None:
        # The mapping is the point: it is what a reader consults instead of
        # guessing which endpoint `stop` became.
        assert LIFECYCLE_OPERATIONS["stop"].startswith("DELETE /runtimes")
        assert LIFECYCLE_OPERATIONS["pause"].endswith("/pause")
        assert LIFECYCLE_OPERATIONS["resume"].endswith("/resume")

    def test_the_base_class_speaks_it(self) -> None:
        for verb in ("supports", "start", "stop", "pause", "resume", "snapshot"):
            assert hasattr(Sandbox, verb), f"Sandbox is missing {verb}"


class TestVariants:
    @pytest.mark.parametrize(
        "module_name, class_name",
        [
            ("code_sandboxes.eval_sandbox", "EvalSandbox"),
            ("code_sandboxes.docker_sandbox", "DockerSandbox"),
            ("code_sandboxes.datalayer_sandbox", "DatalayerSandbox"),
            ("code_sandboxes.jupyter_server_sandbox", "JupyterServerSandbox"),
        ],
    )
    def test_every_variant_conforms(self, module_name: str, class_name: str) -> None:
        import importlib

        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            pytest.skip(f"{module_name} needs an optional dependency: {error}")

        variant = getattr(module, class_name)
        # Structural, not inherited: a class conforms by having the methods.
        for verb in ("supports", "start", "stop", "pause", "resume", "snapshot"):
            assert hasattr(variant, verb), f"{class_name} is missing {verb}"

    def test_a_conforming_instance_satisfies_the_protocol(self) -> None:
        from code_sandboxes.eval_sandbox import EvalSandbox

        assert isinstance(EvalSandbox(), SandboxLifecycle)


class TestRefusals:
    def test_a_provider_says_what_it_cannot_do_before_being_asked(self) -> None:
        from code_sandboxes.eval_sandbox import EvalSandbox

        sandbox = EvalSandbox()

        # "Pause this and come back tomorrow" is a plan a caller makes; it
        # deserves an answer before it commits.
        assert sandbox.supports("start") is True
        assert sandbox.supports("execute") is True
        assert sandbox.supports("pause") is False

    def test_and_refuses_in_the_same_words(self) -> None:
        from code_sandboxes.eval_sandbox import EvalSandbox

        sandbox = EvalSandbox()
        with pytest.raises(SandboxOperationNotSupported) as caught:
            sandbox.pause()

        assert caught.value.operation == "pause"
        assert "supports()" in str(caught.value)

    def test_the_refusal_names_the_variant_when_it_knows_it(self) -> None:
        error = unsupported("snapshot", "docker")

        assert "docker" in str(error)
        assert error.operation == "snapshot"

    def test_an_unknown_verb_is_simply_unsupported(self) -> None:
        from code_sandboxes.eval_sandbox import EvalSandbox

        assert EvalSandbox().supports("teleport") is False


class TestTheTwoShapes:
    """One sandbox and a collection of them are different objects."""

    def test_every_verb_belongs_to_exactly_one_shape(self) -> None:
        from code_sandboxes.lifecycle import (
            INSTANCE_OPERATIONS,
            MANAGER_OPERATIONS,
        )

        assert not INSTANCE_OPERATIONS & MANAGER_OPERATIONS
        assert INSTANCE_OPERATIONS | MANAGER_OPERATIONS == set(LIFECYCLE_OPERATIONS)

    def test_the_instance_protocol_covers_the_instance_verbs(self) -> None:
        from code_sandboxes.lifecycle import INSTANCE_OPERATIONS, SandboxLifecycle

        # "execute" is spelled `run_code` on the protocol, for the same reason
        # `Sandbox` spells it that way: it is the one verb that takes code.
        named = {v for v in INSTANCE_OPERATIONS if v != "execute"}
        assert named <= set(dir(SandboxLifecycle))
        assert "run_code" in dir(SandboxLifecycle)

    def test_the_manager_protocol_covers_the_manager_verbs(self) -> None:
        from code_sandboxes.lifecycle import (
            MANAGER_OPERATIONS,
            SandboxManagerLifecycle,
        )

        assert MANAGER_OPERATIONS <= set(dir(SandboxManagerLifecycle))


class TestTheVocabularyIsReachable:
    """A consumer should not need to know which module it lives in."""

    def test_the_package_root_exports_it(self) -> None:
        import code_sandboxes

        for name in (
            "LIFECYCLE_OPERATIONS",
            "SandboxLifecycle",
            "SandboxManagerLifecycle",
            "SandboxOperationNotSupported",
            "unsupported",
        ):
            assert name in code_sandboxes.__all__
            assert hasattr(code_sandboxes, name)


class TestTheRuntimeUrls:
    """One place knows how a runtime is addressed, so callers cannot drift."""

    BASE = "https://runtimes.example"

    def test_a_trailing_slash_on_the_base_changes_nothing(self) -> None:
        from code_sandboxes.lifecycle import runtime_url

        assert runtime_url(self.BASE, "runtime-1") == runtime_url(self.BASE + "/", "runtime-1")

    def test_the_urls_match_what_the_vocabulary_documents(self) -> None:
        from code_sandboxes.lifecycle import (
            LIFECYCLE_OPERATIONS,
            runtime_pause_url,
            runtime_resume_url,
            runtime_url,
            runtimes_url,
            sandbox_snapshots_url,
        )

        built = {
            "list": runtimes_url(self.BASE),
            "create": runtimes_url(self.BASE),
            "get": runtime_url(self.BASE, "{runtime_name}"),
            "stop": runtime_url(self.BASE, "{runtime_name}"),
            "update": runtime_url(self.BASE, "{runtime_name}"),
            "pause": runtime_pause_url(self.BASE, "{runtime_name}"),
            "resume": runtime_resume_url(self.BASE, "{runtime_name}"),
            "snapshot": sandbox_snapshots_url(self.BASE),
        }
        for verb, url in built.items():
            # The documented spelling is "<METHOD> <path>"; the path is what a
            # URL builder has to agree with.
            path = LIFECYCLE_OPERATIONS[verb].split(" ", 1)[1]
            assert url == f"{self.BASE}/api/runtimes/v1{path}", verb

    def test_stopping_a_paused_runtime_uses_the_same_url(self) -> None:
        from code_sandboxes.lifecycle import runtime_url

        # There is one `stop`, so there is one URL — a paused runtime does not
        # get a second path of its own.
        assert runtime_url(self.BASE, "runtime-1").endswith("/runtimes/runtime-1")
