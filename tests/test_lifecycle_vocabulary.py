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
        except Exception as error:  # noqa: BLE001
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
