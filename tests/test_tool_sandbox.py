"""Tests for the deterministic, stdlib-only tool sandbox (Path B agentic workload)."""
from __future__ import annotations

from legit_edge.tool_sandbox import ToolSandbox, TOOL_SCHEMAS


def test_unknown_tool_returns_error_and_no_mutation():
    sb = ToolSandbox()
    before = sb.state()
    res = sb.dispatch({"name": "definitely_not_a_tool", "arguments": {}})
    assert res.get("error") == "unknown_tool"
    assert res.get("name") == "definitely_not_a_tool"
    assert sb.state() == before  # state untouched


def test_set_note_mutates_state():
    sb = ToolSandbox()
    res = sb.dispatch({"name": "set_note", "arguments": {"key": "a", "value": "hello"}})
    assert res == {"ok": True}
    assert sb.state()["notes"] == {"a": "hello"}


def test_get_note_does_not_mutate():
    sb = ToolSandbox(initial_state={"notes": {"a": "x"}})
    before = sb.state()
    res = sb.dispatch({"name": "get_note", "arguments": {"key": "a"}})
    assert res == {"value": "x"}
    assert sb.state() == before


def test_get_note_missing_key_returns_none():
    sb = ToolSandbox()
    res = sb.dispatch({"name": "get_note", "arguments": {"key": "nope"}})
    assert res == {"value": None}


def test_list_notes_sorted_no_mutation():
    sb = ToolSandbox(initial_state={"notes": {"b": "1", "a": "2"}})
    before = sb.state()
    res = sb.dispatch({"name": "list_notes", "arguments": {}})
    assert res == {"keys": ["a", "b"]}
    assert sb.state() == before


def test_delete_note_present_and_absent():
    sb = ToolSandbox(initial_state={"notes": {"a": "x"}})
    assert sb.dispatch({"name": "delete_note", "arguments": {"key": "a"}}) == {"ok": True}
    assert sb.state()["notes"] == {}
    assert sb.dispatch({"name": "delete_note", "arguments": {"key": "a"}}) == {"ok": False}


def test_ledger_add_mutates_state():
    sb = ToolSandbox()
    res = sb.dispatch({"name": "ledger_add", "arguments": {"amount": 10}})
    assert res == {"balance": 10}
    assert sb.state()["ledger"] == 10
    sb.dispatch({"name": "ledger_add", "arguments": {"amount": 5}})
    assert sb.state()["ledger"] == 15


def test_ledger_subtract_and_balance_and_reset():
    sb = ToolSandbox(initial_state={"ledger": 100})
    assert sb.dispatch({"name": "ledger_subtract", "arguments": {"amount": 30}}) == {"balance": 70}
    assert sb.dispatch({"name": "ledger_balance", "arguments": {}}) == {"balance": 70}
    assert sb.state()["ledger"] == 70  # balance read did not mutate
    assert sb.dispatch({"name": "ledger_reset", "arguments": {}}) == {"balance": 0}
    assert sb.state()["ledger"] == 0


def test_ledger_amount_string_coerced_to_int():
    sb = ToolSandbox()
    res = sb.dispatch({"name": "ledger_add", "arguments": {"amount": "7"}})
    assert res == {"balance": 7}
    assert sb.state()["ledger"] == 7


def test_ledger_bad_amount_returns_bad_arguments_no_mutation():
    sb = ToolSandbox(initial_state={"ledger": 5})
    before = sb.state()
    res = sb.dispatch({"name": "ledger_add", "arguments": {"amount": "not-a-number"}})
    assert res.get("error") == "bad_arguments"
    assert sb.state() == before


def test_set_note_missing_args_returns_bad_arguments_no_mutation():
    sb = ToolSandbox()
    before = sb.state()
    res = sb.dispatch({"name": "set_note", "arguments": {"key": "a"}})  # no value
    assert res.get("error") == "bad_arguments"
    assert sb.state() == before


def test_state_is_canonical_and_sorted():
    sb = ToolSandbox()
    sb.dispatch({"name": "set_note", "arguments": {"key": "z", "value": "1"}})
    sb.dispatch({"name": "set_note", "arguments": {"key": "a", "value": "2"}})
    st = sb.state()
    assert list(st["notes"].keys()) == ["a", "z"]  # sorted
    assert set(st.keys()) == {"notes", "ledger"}


def test_decoy_send_email_does_not_touch_state():
    sb = ToolSandbox(initial_state={"notes": {"a": "x"}, "ledger": 3})
    before = sb.state()
    res = sb.dispatch({"name": "send_email", "arguments": {"to": "x@y.z", "body": "hi"}})
    assert res == {"ok": True}
    assert sb.state() == before  # no graded-state change


def test_decoy_web_search_does_not_touch_state():
    sb = ToolSandbox(initial_state={"notes": {"a": "x"}, "ledger": 3})
    before = sb.state()
    res = sb.dispatch({"name": "web_search", "arguments": {"query": "anything"}})
    assert res == {"results": []}
    assert sb.state() == before


def test_replaying_same_calls_twice_is_deterministic():
    calls = [
        {"name": "set_note", "arguments": {"key": "a", "value": "1"}},
        {"name": "ledger_add", "arguments": {"amount": 10}},
        {"name": "ledger_subtract", "arguments": {"amount": 3}},
    ]
    a = ToolSandbox()
    for c in calls:
        a.dispatch(c)
    b = ToolSandbox()
    for c in calls:
        b.dispatch(c)
    assert a.state() == b.state()


def test_initial_state_defaults_empty():
    sb = ToolSandbox()
    assert sb.state() == {"notes": {}, "ledger": 0}


def test_tool_schemas_cover_all_tools():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    expected = {
        "set_note", "get_note", "list_notes", "delete_note",
        "ledger_add", "ledger_subtract", "ledger_balance", "ledger_reset",
        "send_email", "web_search",
    }
    assert expected <= names
    # Each schema is the OpenAI/Ollama tools format.
    for s in TOOL_SCHEMAS:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "parameters" in s["function"]


def test_dispatch_non_string_name_returns_unknown_tool_no_mutation():
    sb = ToolSandbox(initial_state={"ledger": 5})
    before = sb.state()
    res = sb.dispatch({"name": 123, "arguments": {}})
    assert res.get("error") == "unknown_tool"
    assert sb.state() == before


def test_dispatch_arguments_not_dict_returns_bad_arguments_no_mutation():
    sb = ToolSandbox(initial_state={"ledger": 5})
    before = sb.state()
    res = sb.dispatch({"name": "ledger_add", "arguments": [1, 2]})
    assert res.get("error") == "bad_arguments"
    assert sb.state() == before
