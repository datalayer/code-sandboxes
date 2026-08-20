# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

"""The replay-based session of the Kaggle batch mode."""


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
