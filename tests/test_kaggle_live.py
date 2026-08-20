# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

"""The live Kaggle session: the dataset bus, faked end to end."""

import ast
import json
from pathlib import Path

from code_sandboxes.kaggle_live import KaggleLiveSession, build_agent_code


class FakeApi:
    """Kaggle's dataset API, as a pair of in-memory mailboxes."""

    def __init__(self):
        self.store = {}

    def _absorb(self, folder):
        meta = json.loads((Path(folder) / "dataset-metadata.json").read_text())
        message = json.loads((Path(folder) / "message.json").read_text())
        self.store[meta["id"]] = message

    def dataset_create_new(self, folder, public=False, quiet=True):
        assert public is False, "the bus must be private"
        self._absorb(folder)

    def dataset_create_version(self, folder, version_notes="", quiet=True):
        self._absorb(folder)

    def dataset_download_files(self, ref, path, force=True, unzip=True):
        if ref not in self.store:
            raise FileNotFoundError(ref)
        (Path(path) / "message.json").write_text(json.dumps(self.store[ref]))


class FakeExecutor:
    """The batch executor: records the submission, boots a fake agent."""

    def __init__(self, api):
        self.api = api
        self.submissions = []

    def execute(self, code, **kwargs):
        self.submissions.append({"code": code, **kwargs})
        # The agent's first act is the ready handshake.
        ref = [line for line in code.splitlines() if "k2c" in line]
        assert ref, "the agent must know its outbox"

        class Submitted:
            slug = "user/agent"

        return Submitted()


def _ready(session):
    # Stand in for the booted agent: publish the handshake.
    session._api.store[session._k2c] = {"seq": 0, "reply": {"status": "ok", "outputs": []}}


def test_start_creates_a_private_bus_and_waits_for_the_agent(monkeypatch):
    # The credentials come from the environment of the TEST, never the host's.
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    api = FakeApi()
    session = KaggleLiveSession(FakeExecutor(api), api=api, poll_seconds=0.01)
    # The handshake arrives while start() polls: plant it up front by hooking
    # the executor's submission, which happens before the wait.
    original = session._executor.execute

    def execute(code, **kwargs):
        result = original(code, **kwargs)
        _ready(session)
        return result

    session._executor.execute = execute
    session.start(ready_timeout=1)
    assert session._c2k.endswith("-c2k") and session._k2c.endswith("-k2c")
    assert api.store[session._c2k] == {"seq": 0}


def test_execute_round_trips_over_the_bus():
    api = FakeApi()
    session = KaggleLiveSession(FakeExecutor(api), api=api, poll_seconds=0.01)
    session._c2k = "user/bus-c2k"
    session._k2c = "user/bus-k2c"
    api.store[session._c2k] = {"seq": 0}

    real_read = session._read_bus

    def read(ref):
        # The fake agent answers the moment it sees the new sequence.
        inbox = api.store.get(session._c2k, {})
        if inbox.get("seq") == session._seq:
            api.store[session._k2c] = {
                "seq": session._seq,
                "reply": {"status": "ok", "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": "42\n"}
                ]},
            }
        return real_read(ref)

    session._read_bus = read
    reply = session.execute("print(42)", timeout=1)
    assert reply["status"] == "ok"
    assert reply["outputs"][0]["text"] == "42\n"


def test_the_generated_agent_is_valid_python():
    agent = build_agent_code(
        {"KAGGLE_USERNAME": "user", "KAGGLE_KEY": "key"},
        "user/a-c2k",
        "user/a-k2c",
    )
    ast.parse(agent)
    assert "user/a-c2k" in agent and "user/a-k2c" in agent
    # The credentials ride in the settings blob, nowhere else.
    assert "KAGGLE_KEY" in agent


def test_the_agent_carries_an_access_token_verbatim():
    # A KGAT access token authenticates the agent's kaggle client through
    # the same variable it was given in — no username/key pair involved.
    agent = build_agent_code(
        {"KAGGLE_API_TOKEN": "KGAT_abc123"},
        "user/a-c2k",
        "user/a-k2c",
    )
    ast.parse(agent)
    assert "KGAT_abc123" in agent
    assert "KAGGLE_KEY" not in agent
