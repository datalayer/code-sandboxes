# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

"""The replay-based session of the Kaggle batch mode."""

from typing import ClassVar


class TestKaggleBatchSession:
    """The batch session: state carried by replaying the code that ran."""

    def _sandbox(self, **kwargs):
        from code_sandboxes.kaggle_sandbox import KaggleSandbox

        return KaggleSandbox(**kwargs)

    def test_prelude_replays_history_behind_a_sentinel(self):
        sandbox = self._sandbox()
        sandbox._session_history = ["x = 1"]
        full, marker = sandbox._session_prelude("print(x)")
        assert marker is not None
        assert full.startswith("x = 1\n")
        assert full.endswith("\nprint(x)")
        assert repr(marker) in full

    def test_prelude_is_plain_on_the_first_turn_and_when_off(self):
        sandbox = self._sandbox()
        assert sandbox._session_prelude("x = 1") == ("x = 1", None)
        off = self._sandbox(session=False)
        off._session_history = ["x = 1"]
        assert off._session_prelude("print(x)") == ("print(x)", None)

    def test_replayed_outputs_are_cut_at_the_sentinel(self):
        sandbox = self._sandbox()
        marker = "<<code-sandbox-session-abc>>"
        reply = {
            "outputs": [
                {"output_type": "stream", "name": "stdout", "text": "old\n"},
                {"output_type": "stream", "name": "stdout", "text": f"{marker}\nnew\n"},
                {"output_type": "execute_result", "data": {"text/plain": "2"}},
            ]
        }
        cut = sandbox._cut_replayed_outputs(reply, marker)
        texts = [o.get("text") for o in cut["outputs"] if o["output_type"] == "stream"]
        assert texts == ["new\n"]
        assert cut["outputs"][-1]["output_type"] == "execute_result"

    def test_missing_sentinel_keeps_everything(self):
        sandbox = self._sandbox()
        reply = {"outputs": [{"output_type": "stream", "name": "stdout", "text": "boom"}]}
        assert sandbox._cut_replayed_outputs(reply, "<<never>>") == reply


class TestKaggleStreamingSession:
    """What the STREAMING path records, which later turns re-run.

    The two batch paths have to agree: what one records, the other records,
    and on the same condition. Streaming judged by the status of the JOB —
    and a job whose cell raised still completes, so a failing snippet joined
    the replay and failed every turn after it.
    """

    def _sandbox(self, outputs, status="COMPLETE"):
        from types import SimpleNamespace

        from code_sandboxes.kaggle_sandbox import KaggleSandbox

        class _Executor:
            """No `api` and no `output`: no polling, no artifact download."""

            def __init__(self):
                self.submitted = []

            def execute(self, code, **_kwargs):
                self.submitted.append(code)
                return SimpleNamespace(
                    slug="user/job",
                    status=status,
                    log="",
                    to_kernel_reply=lambda: {"outputs": outputs},
                )

        sandbox = KaggleSandbox()
        sandbox._batch_mode = True
        sandbox._executor = _Executor()
        return sandbox

    _PRINTED: ClassVar[list] = [{"output_type": "stream", "name": "stdout", "text": "ok\n"}]
    _RAISED: ClassVar[list] = [
        {"output_type": "error", "ename": "NameError", "evalue": "x", "traceback": []}
    ]

    def test_a_snippet_that_worked_joins_the_session(self):
        sandbox = self._sandbox(self._PRINTED)

        list(sandbox.run_code_streaming("x = 1"))

        assert sandbox._session_history == ["x = 1"]

    def test_a_snippet_that_raised_does_not_join_the_session(self):
        """The job completed; the code did not. Only the second one counts."""
        sandbox = self._sandbox(self._RAISED)

        list(sandbox.run_code_streaming("print(x)"))

        assert sandbox._session_history == []

    def test_a_job_that_failed_does_not_join_the_session(self):
        sandbox = self._sandbox(self._PRINTED, status="ERROR")

        list(sandbox.run_code_streaming("x = 1"))

        assert sandbox._session_history == []

    def test_the_next_turn_does_not_replay_what_failed(self):
        """Why it matters: the replay runs ahead of every later snippet."""
        sandbox = self._sandbox(self._RAISED)

        list(sandbox.run_code_streaming("print(x)"))

        assert sandbox._session_prelude("y = 2") == ("y = 2", None)
