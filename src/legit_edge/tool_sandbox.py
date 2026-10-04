"""Deterministic, stdlib-only stateful tool sandbox for the multi-step agentic workload.

No eval/exec, no external deps. The model's tool calls are dispatched by name to a
fixed WHITELIST of Python callables over two graded domains — a notes key/value store
and a numeric ledger — plus a few benign decoy tools used to probe missing-function
handling. State is integer-exact (ledger) and string-keyed (notes) so equality of the
canonical ``state()`` snapshot is an exact oracle for the state-diff scoring rule.

Shared by the live per-turn agent loop (runner) and by ``MultiTurnToolWorkload`` replay.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable


class ToolSandbox:
    """A small, deterministic, whitelisted tool environment.

    State:
        notes: dict[str, str]   key/value note store
        ledger: int             integer running balance
    """

    def __init__(self, initial_state: dict | None = None) -> None:
        initial_state = initial_state or {}
        notes = initial_state.get("notes") or {}
        self.notes: dict[str, str] = {str(k): str(v) for k, v in dict(notes).items()}
        self.ledger: int = int(initial_state.get("ledger", 0) or 0)
        # WHITELIST dispatch table: name -> bound handler taking the arguments dict.
        self._tools: dict[str, Callable[[dict], dict]] = {
            "set_note": self._set_note,
            "get_note": self._get_note,
            "list_notes": self._list_notes,
            "delete_note": self._delete_note,
            "ledger_add": self._ledger_add,
            "ledger_subtract": self._ledger_subtract,
            "ledger_balance": self._ledger_balance,
            "ledger_reset": self._ledger_reset,
            # Decoys: in the whitelist (so they don't error) but never touch graded state.
            "send_email": self._send_email,
            "web_search": self._web_search,
        }

    # ------------------------------------------------------------------ dispatch
    def dispatch(self, call: dict) -> dict:
        """Execute one tool call ``{"name": str, "arguments": dict}``.

        Unknown tool name -> ``{"error": "unknown_tool", "name": name}`` (no mutation).
        Bad/missing arguments -> ``{"error": "bad_arguments", ...}`` (no mutation).
        Never eval/exec.
        """
        name = call.get("name")
        handler = self._tools.get(name) if isinstance(name, str) else None
        if handler is None:
            return {"error": "unknown_tool", "name": name}
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            return {"error": "bad_arguments", "name": name, "reason": "arguments_not_dict"}
        return handler(args)

    def state(self) -> dict:
        """Canonical snapshot for state-diff comparison (notes sorted by key)."""
        return {"notes": dict(sorted(self.notes.items())), "ledger": self.ledger}

    # ----------------------------------------------------------- notes tools
    def _set_note(self, args: dict) -> dict:
        if "key" not in args or "value" not in args:
            return {"error": "bad_arguments", "name": "set_note"}
        self.notes[str(args["key"])] = str(args["value"])
        return {"ok": True}

    def _get_note(self, args: dict) -> dict:
        if "key" not in args:
            return {"error": "bad_arguments", "name": "get_note"}
        return {"value": self.notes.get(str(args["key"]))}

    def _list_notes(self, args: dict) -> dict:
        return {"keys": sorted(self.notes)}

    def _delete_note(self, args: dict) -> dict:
        if "key" not in args:
            return {"error": "bad_arguments", "name": "delete_note"}
        key = str(args["key"])
        was_present = key in self.notes
        if was_present:
            del self.notes[key]
        return {"ok": was_present}

    # ---------------------------------------------------------- ledger tools
    def _coerce_amount(self, args: dict) -> int | None:
        if "amount" not in args:
            return None
        try:
            return int(args["amount"])
        except (TypeError, ValueError):
            return None

    def _ledger_add(self, args: dict) -> dict:
        amount = self._coerce_amount(args)
        if amount is None:
            return {"error": "bad_arguments", "name": "ledger_add"}
        self.ledger += amount
        return {"balance": self.ledger}

    def _ledger_subtract(self, args: dict) -> dict:
        amount = self._coerce_amount(args)
        if amount is None:
            return {"error": "bad_arguments", "name": "ledger_subtract"}
        self.ledger -= amount
        return {"balance": self.ledger}

    def _ledger_balance(self, args: dict) -> dict:
        return {"balance": self.ledger}

    def _ledger_reset(self, args: dict) -> dict:
        self.ledger = 0
        return {"balance": 0}

    # ------------------------------------------------------------- decoys
    def _send_email(self, args: dict) -> dict:
        # Benign no-op: does NOT touch notes/ledger.
        return {"ok": True}

    def _web_search(self, args: dict) -> dict:
        # Benign no-op: does NOT touch notes/ledger.
        return {"results": []}


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# Function schemas (OpenAI/Ollama tools format) for ALL tools (real + decoys),
# exported for the runner to present to the model.
TOOL_SCHEMAS: list[dict] = [
    _fn(
        "set_note",
        "Store a note under a key in the key/value note store.",
        {
            "key": {"type": "string", "description": "Note key."},
            "value": {"type": "string", "description": "Note value to store."},
        },
        ["key", "value"],
    ),
    _fn(
        "get_note",
        "Read the value of a note by key. Returns null if the key is absent.",
        {"key": {"type": "string", "description": "Note key to read."}},
        ["key"],
    ),
    _fn(
        "list_notes",
        "List all note keys, sorted.",
        {},
        [],
    ),
    _fn(
        "delete_note",
        "Delete a note by key. Returns whether the key was present.",
        {"key": {"type": "string", "description": "Note key to delete."}},
        ["key"],
    ),
    _fn(
        "ledger_add",
        "Add an integer amount to the running ledger balance.",
        {"amount": {"type": "integer", "description": "Integer amount to add."}},
        ["amount"],
    ),
    _fn(
        "ledger_subtract",
        "Subtract an integer amount from the running ledger balance.",
        {"amount": {"type": "integer", "description": "Integer amount to subtract."}},
        ["amount"],
    ),
    _fn(
        "ledger_balance",
        "Read the current ledger balance.",
        {},
        [],
    ),
    _fn(
        "ledger_reset",
        "Reset the ledger balance to zero.",
        {},
        [],
    ),
    # Decoys (distractors; benign no-ops).
    _fn(
        "send_email",
        "Send an email (decoy / irrelevant to the notes and ledger state).",
        {
            "to": {"type": "string", "description": "Recipient address."},
            "body": {"type": "string", "description": "Email body."},
        },
        ["to", "body"],
    ),
    _fn(
        "web_search",
        "Search the web (decoy / irrelevant to the notes and ledger state).",
        {"query": {"type": "string", "description": "Search query."}},
        ["query"],
    ),
]


def _normalize_call(entry: Any) -> dict | None:
    """Coerce one parsed list entry into a ``{"name","arguments"}`` call, or None.

    Well-formed = a dict with a string ``name``. ``arguments`` defaults to ``{}`` and
    is dropped (treated as ``{}``) if not a dict, so the sandbox's own bad-argument
    guard is the single source of truth for argument validation.
    """
    if not isinstance(entry, dict):
        return None
    name = entry.get("name")
    if not isinstance(name, str):
        return None
    args = entry.get("arguments", {})
    if not isinstance(args, dict):
        args = {}
    return {"name": name, "arguments": args}


def parse_tool_calls(text: str) -> list[dict]:
    """Extract a JSON array of ``{"name","arguments"}`` tool calls from model text.

    The model is asked to emit ONLY a JSON array, but we tolerate surrounding prose
    and ``` code fences. Strategy: scan for each balanced top-level ``[...]`` span and
    try ``json.loads``; the first span that parses to a list containing at least one
    well-formed call wins. Malformed entries within that list are dropped. Returns
    ``[]`` if nothing parses (the runner reads ``[]`` as "the model gave a final answer").
    """
    if not text:
        return []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    span = text[start : i + 1]
                    try:
                        parsed = json.loads(span)
                    except (ValueError, TypeError):
                        start = -1
                        continue
                    if isinstance(parsed, list):
                        calls = [
                            c for c in (_normalize_call(e) for e in parsed)
                            if c is not None
                        ]
                        if calls:
                            return calls
                    start = -1
    return []


def build_tool_system_prompt() -> str:
    """Deterministic ReAct system preamble: tool catalogue + the JSON-array protocol.

    Lists each tool's name, one-line description, and JSON parameters (derived from
    ``TOOL_SCHEMAS`` in declared order — no timestamps, so the prompt is stable across
    runs and the experiment is reproducible).
    """
    lines = [
        "You are a careful tool-using assistant operating over a notes key/value store "
        "and an integer ledger.",
        "",
        "Available tools:",
    ]
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        name = fn["name"]
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        params_json = json.dumps(params, sort_keys=True)
        lines.append(f"- {name}: {desc} params={params_json}")
    lines += [
        "",
        "PROTOCOL:",
        "To use tools, respond with ONLY a JSON array of calls like "
        '[{"name": "tool_name", "arguments": {...}}]. You may call several tools at '
        "once by listing them in the array; their results will be returned to you.",
        "When the task for the current turn is complete, respond with a short plain-text "
        "confirmation and NO JSON array.",
    ]
    return "\n".join(lines)
