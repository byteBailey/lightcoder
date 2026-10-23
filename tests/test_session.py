"""Append-only session transcripts: write, read back, resume.

The contract: every message that enters the conversation is written as it arrives (so an
interrupted run keeps its history), a resumed session continues the *same* file without
duplicating what it replayed, and a run with no session attached behaves exactly as before.
"""

import json

import pytest

from lightcoder import Agent, AgentConfig
from lightcoder.session import SessionLog, describe


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTCODER_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    return tmp_path / "home"


# ------------------------------------------------------------------ write/read

def test_create_writes_header_and_messages(home):
    log = SessionLog.create(model="m1", config="baseline")
    log.message({"role": "user", "content": "hello"})
    log.message({"role": "assistant", "content": "hi"})

    header, messages = SessionLog.read(log.path)
    assert header["type"] == "session" and header["model"] == "m1"
    assert header["id"] == log.id
    assert [m["content"] for m in messages] == ["hello", "hi"]


def test_each_record_is_its_own_line(home):
    log = SessionLog.create()
    log.message({"role": "user", "content": "a"})
    lines = log.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2                      # header + one message
    assert all(json.loads(ln) for ln in lines)  # every line stands alone


def test_partial_final_line_is_skipped(home):
    """A run killed mid-write leaves a truncated line; reading must not fail."""
    log = SessionLog.create()
    log.message({"role": "user", "content": "kept"})
    with open(log.path, "a", encoding="utf-8") as f:
        f.write('{"type": "message", "message": {"role": "user", "cont')  # torn write

    _, messages = SessionLog.read(log.path)
    assert [m["content"] for m in messages] == ["kept"]


# ------------------------------------------------------------------ lookup

def test_list_latest_and_find(home):
    a = SessionLog.create()
    b = SessionLog.create()
    assert SessionLog.list() == sorted([a.path, b.path])
    assert SessionLog.latest() == max(a.path, b.path)
    assert SessionLog.find(b.id) == b.path
    assert SessionLog.find(b.id[-6:]) == b.path      # id suffix
    assert SessionLog.find("nope-nope") is None


def test_describe_summarises(home):
    log = SessionLog.create(model="some-model")
    log.message({"role": "user", "content": "fix the parser bug"})
    line = describe(log.path)
    assert log.id in line and "1 msgs" in line and "fix the parser bug" in line


# ------------------------------------------------------------------ agent wiring

def _agent(session=None):
    a = Agent(None, config=AgentConfig(compression=False, subagent=False))
    a.session = session
    return a


def test_agent_records_into_transcript(home):
    log = SessionLog.create()
    a = _agent(log)
    a._record({"role": "user", "content": "q"})
    a._record({"role": "assistant", "content": "r"})

    _, messages = SessionLog.read(log.path)
    assert [m["content"] for m in messages] == ["q", "r"]
    assert len(a.messages) == 2          # live history still maintained


def test_agent_without_session_is_unaffected(home):
    a = _agent(None)
    a._record({"role": "user", "content": "q"})
    assert a.messages == [{"role": "user", "content": "q"}]
    assert SessionLog.list() == []       # nothing written


# ------------------------------------------------------------------ resume

class _Args:
    def __init__(self, **kw):
        self.no_session = kw.get("no_session", False)
        self.resume = kw.get("resume", None)


def test_resume_replays_without_duplicating(home):
    from lightcoder.cli import _attach_session

    log = SessionLog.create(model="m")
    log.message({"role": "user", "content": "first"})
    log.message({"role": "assistant", "content": "second"})

    a = _agent(None)
    _attach_session(a, _Args(resume=""))          # bare --resume: most recent

    assert [m["content"] for m in a.messages] == ["first", "second"]  # replayed
    _, on_disk = SessionLog.read(log.path)
    assert len(on_disk) == 2                       # NOT written back to the file
    assert a.session is not None and a.session.path == log.path       # same file continues


def test_resume_then_new_message_appends_to_same_file(home):
    from lightcoder.cli import _attach_session

    log = SessionLog.create()
    log.message({"role": "user", "content": "old"})

    a = _agent(None)
    _attach_session(a, _Args(resume=""))
    a._record({"role": "user", "content": "new"})

    _, messages = SessionLog.read(log.path)
    assert [m["content"] for m in messages] == ["old", "new"]
    assert len(SessionLog.list()) == 1             # no second file created


def test_resume_with_no_prior_session_starts_fresh(home):
    from lightcoder.cli import _attach_session

    a = _agent(None)
    _attach_session(a, _Args(resume=""))
    assert a.messages == [] and a.session is not None


def test_no_session_flag_disables_logging(home):
    from lightcoder.cli import _attach_session

    a = _agent(None)
    _attach_session(a, _Args(no_session=True))
    assert a.session is None and SessionLog.list() == []
