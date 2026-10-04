"""Deterministic generator for the bespoke multi-step tool-use tasks (Path B agentic workload).

This module authors the TASK DATA for ``MultiTurnToolWorkload``. Each task is a multi-turn
(2-4 user turns) tool-use scenario with **state-dependent** turns: a later turn reads or
modifies state that a prior turn established. The graded toolset is exactly the
``ToolSandbox`` whitelist — a notes key/value store (``set_note``/``get_note``/``list_notes``/
``delete_note``), an integer ledger (``ledger_add``/``ledger_subtract``/``ledger_balance``/
``ledger_reset``), plus two benign decoys (``send_email``/``web_search``) used to probe
missing-function handling.

The oracle for every task is a **state-diff**: each row carries ``expected_states[i]`` =
the canonical ``ToolSandbox.state()`` AFTER user turn ``i``. These are NEVER hand-written —
``generate_multiturn_tasks`` computes them by REPLAYING a reference call sequence
(``reference_calls``: one list of calls per turn) through a fresh ``ToolSandbox`` seeded from
the task's ``initial_state``. This makes the pinned oracle achievable-by-construction for a
correct agent.

Sub-cases mirror BFCL multi-turn:
  * ``base`` — every turn is in-scope and performs the obvious state change.
  * ``missing_function`` — one turn asks for an out-of-scope capability no whitelisted tool
    provides; the correct behaviour is to make NO state-changing call, so that turn's
    ``reference_calls`` is ``[]`` and its expected state equals the prior state.
  * ``missing_parameter`` — one turn omits a value required to act, so the correct behaviour
    is no change that turn (``reference_calls == []``, state unchanged); a LATER turn supplies
    the value and the action then executes.

Determinism: ``generate_multiturn_tasks(seed=42)`` seeds a single ``random.Random`` and
instantiates a fixed, ordered template schedule, so the pool is byte-stable across runs.
All note keys/values are kept free of ``[`` and ``]`` because the runner's tool-call parser
is bracket-counting.
"""
from __future__ import annotations

import random
from typing import Callable

from .tool_sandbox import ToolSandbox


# --------------------------------------------------------------------------- value pools
# All entries are bracket-free (the runner's call parser counts '[' / ']').
_PAYEES = [
    "alice", "bob", "carlos", "dana", "erin", "frank", "grace", "hiro",
    "ivy", "jamal", "kira", "leo", "mona", "nina", "omar", "priya",
]
_VENDORS = [
    "acme-supply", "globex", "initech", "umbrella", "wayne-co", "stark-ind",
    "wonka", "tyrell", "soylent", "hooli", "pied-piper", "dunder-mifflin",
]
_PROJECTS = [
    "atlas", "borealis", "comet", "delta", "echo", "falcon", "gemini",
    "horizon", "ion", "juno", "kepler", "lyra",
]
_CITIES = [
    "Paris", "Tokyo", "Cairo", "Lima", "Oslo", "Accra", "Hanoi", "Quito",
]
_FRUITS = ["apples", "pears", "mangoes", "grapes", "plums", "kiwis", "figs", "limes"]


def _call(name: str, **arguments) -> dict:
    """Build one ``{"name","arguments"}`` tool call (matches ToolSandbox.dispatch input)."""
    return {"name": name, "arguments": dict(arguments)}


# A "Task" produced by a template: the reference_calls drive the replay oracle.
# {id?, sub_type, turns: list[str], initial_state: dict, reference_calls: list[list[dict]]}
Template = Callable[[random.Random, str], dict]


# --------------------------------------------------------------------------- templates
# Each template accepts (rng, sub_type) and returns a task WITHOUT an id (assigned later).
# sub_type is one of {"base","missing_function","missing_parameter"}. Templates that can
# host an out-of-scope or no-value turn implement those branches; "base" is always the
# straightforward happy path. reference_calls for a no-change turn is [].


def t_budget(rng: random.Random, sub_type: str) -> dict:
    """Budgeting flow: open a ledger, then spend and query against it."""
    vendor = rng.choice(_VENDORS)
    seed = rng.choice([100, 150, 200, 250, 300])
    spend = rng.choice([20, 35, 40, 55, 70])
    if sub_type == "missing_function":
        # Turn 3 asks to convert the balance to another currency — no tool does FX.
        turns = [
            f"Start a budget for {vendor} by adding {seed} to the ledger.",
            f"We paid {vendor} {spend}, subtract that from the budget.",
            "Now convert the remaining balance into euros for me.",
        ]
        ref = [
            [_call("ledger_add", amount=seed)],
            [_call("ledger_subtract", amount=spend)],
            [],  # FX is out of scope -> no graded state change
        ]
    elif sub_type == "missing_parameter":
        # Turn 2 says "subtract the invoice" without the amount; turn 3 supplies it.
        turns = [
            f"Start a budget for {vendor} by adding {seed} to the ledger.",
            f"Subtract the {vendor} invoice from the budget.",
            f"The {vendor} invoice was {spend}; go ahead and subtract it.",
        ]
        ref = [
            [_call("ledger_add", amount=seed)],
            [],  # amount unknown -> no change this turn
            [_call("ledger_subtract", amount=spend)],
        ]
    else:  # base
        topup = rng.choice([10, 25, 60])
        turns = [
            f"Start a budget for {vendor} by adding {seed} to the ledger.",
            f"We paid {vendor} {spend}, subtract that from the budget.",
            f"Add a top-up of {topup} to the budget.",
        ]
        ref = [
            [_call("ledger_add", amount=seed)],
            [_call("ledger_subtract", amount=spend)],
            [_call("ledger_add", amount=topup)],
        ]
    return {
        "sub_type": sub_type,
        "turns": turns,
        "initial_state": {"notes": {}, "ledger": 0},
        "reference_calls": ref,
    }


def t_contacts(rng: random.Random, sub_type: str) -> dict:
    """Contacts notebook: store a contact, update it, then list (state-dependent)."""
    person = rng.choice(_PAYEES)
    city1 = rng.choice(_CITIES)
    city2 = rng.choice([c for c in _CITIES if c != city1])
    if sub_type == "missing_function":
        turns = [
            f"Save a contact note: key {person}, value {city1}.",
            f"Actually {person} moved, update the note value to {city2}.",
            f"Send {person} a text message letting them know we updated it.",
        ]
        ref = [
            [_call("set_note", key=person, value=city1)],
            [_call("set_note", key=person, value=city2)],
            [],  # texting is out of scope (only send_email decoy exists, ungraded)
        ]
    elif sub_type == "missing_parameter":
        turns = [
            f"Save a contact note for {person}.",  # value omitted
            f"The value for {person} should be {city1}; save it.",
            f"{person} moved, change the note value to {city2}.",
        ]
        ref = [
            [],  # no value -> cannot store yet
            [_call("set_note", key=person, value=city1)],
            [_call("set_note", key=person, value=city2)],
        ]
    else:  # base
        turns = [
            f"Save a contact note: key {person}, value {city1}.",
            f"{person} moved, update the note value to {city2}.",
            "List all the contact note keys you have.",
        ]
        ref = [
            [_call("set_note", key=person, value=city1)],
            [_call("set_note", key=person, value=city2)],
            [_call("list_notes")],  # read-only: no state change, but a real in-scope action
        ]
    return {
        "sub_type": sub_type,
        "turns": turns,
        "initial_state": {"notes": {}, "ledger": 0},
        "reference_calls": ref,
    }


def t_note_then_ledger(rng: random.Random, sub_type: str) -> dict:
    """Store a numeric value as a note, then drive the ledger using that value."""
    project = rng.choice(_PROJECTS)
    amount = rng.choice([12, 18, 24, 30, 45])
    if sub_type == "missing_function":
        turns = [
            f"Record a note: key {project}-cost, value {amount}.",
            f"Add the {project}-cost amount ({amount}) to the ledger.",
            f"Email the {project} stakeholders a summary of the cost.",
        ]
        ref = [
            [_call("set_note", key=f"{project}-cost", value=str(amount))],
            [_call("ledger_add", amount=amount)],
            [],  # emailing stakeholders is out of scope for graded state
        ]
    elif sub_type == "missing_parameter":
        turns = [
            f"Record a note: key {project}-cost, value {amount}.",
            f"Add the {project}-cost amount to the ledger.",  # relies on the stored value
            f"Use {amount} as the {project}-cost and add it to the ledger.",
        ]
        ref = [
            [_call("set_note", key=f"{project}-cost", value=str(amount))],
            [],  # amount not restated explicitly -> treat as unresolved this turn
            [_call("ledger_add", amount=amount)],
        ]
    else:  # base
        extra = rng.choice([5, 8, 15])
        turns = [
            f"Record a note: key {project}-cost, value {amount}.",
            f"Add the {project}-cost amount ({amount}) to the ledger.",
            f"Then add another {extra} for shipping.",
        ]
        ref = [
            [_call("set_note", key=f"{project}-cost", value=str(amount))],
            [_call("ledger_add", amount=amount)],
            [_call("ledger_add", amount=extra)],
        ]
    return {
        "sub_type": sub_type,
        "turns": turns,
        "initial_state": {"notes": {}, "ledger": 0},
        "reference_calls": ref,
    }


def t_reconcile(rng: random.Random, sub_type: str) -> dict:
    """Reconcile a pre-funded ledger: subtract several charges (initial_state non-empty)."""
    start = rng.choice([200, 300, 500])
    c1 = rng.choice([30, 45, 60])
    c2 = rng.choice([15, 25, 50])
    if sub_type == "missing_function":
        turns = [
            f"Our account starts at {start}. Subtract a charge of {c1}.",
            f"Subtract another charge of {c2}.",
            "Print a paper receipt of the final balance.",
        ]
        ref = [
            [_call("ledger_subtract", amount=c1)],
            [_call("ledger_subtract", amount=c2)],
            [],  # printing is out of scope
        ]
    elif sub_type == "missing_parameter":
        turns = [
            f"Our account starts at {start}. Subtract a charge of {c1}.",
            "Subtract the second charge as well.",  # amount omitted
            f"The second charge is {c2}; subtract it.",
        ]
        ref = [
            [_call("ledger_subtract", amount=c1)],
            [],  # unknown amount -> no change
            [_call("ledger_subtract", amount=c2)],
        ]
    else:  # base
        turns = [
            f"Our account starts at {start}. Subtract a charge of {c1}.",
            f"Subtract another charge of {c2}.",
            "What's the current balance?",
        ]
        ref = [
            [_call("ledger_subtract", amount=c1)],
            [_call("ledger_subtract", amount=c2)],
            [_call("ledger_balance")],  # read-only in-scope action
        ]
    return {
        "sub_type": sub_type,
        "turns": turns,
        "initial_state": {"notes": {}, "ledger": start},
        "reference_calls": ref,
    }


def t_inventory(rng: random.Random, sub_type: str) -> dict:
    """Inventory notebook: store a count, then delete a discontinued item."""
    fruit = rng.choice(_FRUITS)
    fruit2 = rng.choice([f for f in _FRUITS if f != fruit])
    count = rng.choice([6, 9, 12, 20])
    if sub_type == "missing_function":
        turns = [
            f"Note the inventory: key {fruit}, value {count}.",
            f"We discontinued {fruit2}, delete its note if present.",
            f"Order more {fruit} from the supplier automatically.",
        ]
        ref = [
            [_call("set_note", key=fruit, value=str(count))],
            [_call("delete_note", key=fruit2)],  # idempotent even if absent
            [],  # auto-ordering is out of scope
        ]
    elif sub_type == "missing_parameter":
        turns = [
            f"Note the inventory: key {fruit}, value {count}.",
            "Update the count for it.",  # which value? omitted
            f"Set the {fruit} count to {count + 5}.",
        ]
        ref = [
            [_call("set_note", key=fruit, value=str(count))],
            [],  # no new value -> no change
            [_call("set_note", key=fruit, value=str(count + 5))],
        ]
    else:  # base
        turns = [
            f"Note the inventory: key {fruit}, value {count}.",
            f"Also note key {fruit2}, value {count + 3}.",
            f"We discontinued {fruit}, delete its note.",
        ]
        ref = [
            [_call("set_note", key=fruit, value=str(count))],
            [_call("set_note", key=fruit2, value=str(count + 3))],
            [_call("delete_note", key=fruit)],
        ]
    return {
        "sub_type": sub_type,
        "turns": turns,
        "initial_state": {"notes": {}, "ledger": 0},
        "reference_calls": ref,
    }


def t_transfer(rng: random.Random, sub_type: str) -> dict:
    """Two-turn transfer: add funds, then move part out (compact state-dependent flow)."""
    payee = rng.choice(_PAYEES)
    add = rng.choice([80, 120, 160])
    out = rng.choice([20, 40, 60])
    if sub_type == "missing_function":
        turns = [
            f"Credit the ledger with {add} for {payee}.",
            f"Wire {out} from this balance to {payee}'s external bank.",
        ]
        # Wiring externally is out of scope; the in-scope effect is just the credit.
        ref = [
            [_call("ledger_add", amount=add)],
            [],  # external wire not supported -> no graded change
        ]
    elif sub_type == "missing_parameter":
        turns = [
            f"Credit the ledger with {add} for {payee}.",
            "Now subtract the transfer amount.",  # amount omitted
            f"The transfer to {payee} is {out}; subtract it.",
        ]
        ref = [
            [_call("ledger_add", amount=add)],
            [],
            [_call("ledger_subtract", amount=out)],
        ]
    else:  # base
        turns = [
            f"Credit the ledger with {add} for {payee}.",
            f"Now subtract a transfer of {out} to {payee}.",
        ]
        ref = [
            [_call("ledger_add", amount=add)],
            [_call("ledger_subtract", amount=out)],
        ]
    return {
        "sub_type": sub_type,
        "turns": turns,
        "initial_state": {"notes": {}, "ledger": 0},
        "reference_calls": ref,
    }


def t_reset_rebuild(rng: random.Random, sub_type: str) -> dict:
    """Reset a funded ledger to zero, then rebuild it (state-dependent on the reset)."""
    start = rng.choice([90, 140, 210])
    rebuild = rng.choice([50, 75, 100])
    if sub_type == "missing_function":
        turns = [
            f"The ledger holds {start}. Reset it to zero to start fresh.",
            f"Add {rebuild} as the new opening balance.",
            "Schedule a monthly auto-deposit going forward.",
        ]
        ref = [
            [_call("ledger_reset")],
            [_call("ledger_add", amount=rebuild)],
            [],  # scheduling recurring deposits is out of scope
        ]
    elif sub_type == "missing_parameter":
        turns = [
            f"The ledger holds {start}. Reset it to zero to start fresh.",
            "Add the new opening balance.",  # amount omitted
            f"Use {rebuild} as the opening balance and add it.",
        ]
        ref = [
            [_call("ledger_reset")],
            [],
            [_call("ledger_add", amount=rebuild)],
        ]
    else:  # base
        turns = [
            f"The ledger holds {start}. Reset it to zero to start fresh.",
            f"Add {rebuild} as the new opening balance.",
            "Report the balance.",
        ]
        ref = [
            [_call("ledger_reset")],
            [_call("ledger_add", amount=rebuild)],
            [_call("ledger_balance")],
        ]
    return {
        "sub_type": sub_type,
        "turns": turns,
        "initial_state": {"notes": {}, "ledger": start},
        "reference_calls": ref,
    }


def t_notes_cleanup(rng: random.Random, sub_type: str) -> dict:
    """Notes housekeeping starting from a PRE-POPULATED store: add, then delete one."""
    keep = rng.choice(_PROJECTS)
    drop = rng.choice([p for p in _PROJECTS if p != keep])
    add_key = rng.choice([p for p in _PROJECTS if p not in (keep, drop)])
    val = rng.choice(["active", "paused", "review", "shipped"])
    initial_notes = {keep: "active", drop: "stale"}
    if sub_type == "missing_function":
        turns = [
            f"Add a note: key {add_key}, value {val}.",
            f"Archive the {drop} note to cold storage.",  # no archive tool
            f"Delete the {drop} note.",
        ]
        ref = [
            [_call("set_note", key=add_key, value=val)],
            [],  # "archive to cold storage" is out of scope -> no change
            [_call("delete_note", key=drop)],
        ]
    elif sub_type == "missing_parameter":
        turns = [
            "Add a new project note.",  # key and value omitted
            f"Make it key {add_key}, value {val}.",
            f"Now delete the {drop} note.",
        ]
        ref = [
            [],  # nothing to add yet
            [_call("set_note", key=add_key, value=val)],
            [_call("delete_note", key=drop)],
        ]
    else:  # base
        turns = [
            f"Add a note: key {add_key}, value {val}.",
            f"Delete the {drop} note.",
            "List the remaining note keys.",
        ]
        ref = [
            [_call("set_note", key=add_key, value=val)],
            [_call("delete_note", key=drop)],
            [_call("list_notes")],
        ]
    return {
        "sub_type": sub_type,
        "turns": turns,
        "initial_state": {"notes": dict(initial_notes), "ledger": 0},
        "reference_calls": ref,
    }


# Ordered template registry (8 templates).
_TEMPLATES: list[Template] = [
    t_budget,
    t_contacts,
    t_note_then_ledger,
    t_reconcile,
    t_inventory,
    t_transfer,
    t_reset_rebuild,
    t_notes_cleanup,
]


def _subtype_schedule(n: int) -> list[str]:
    """Fixed sub_type plan for the pool: base is the plurality, others >= 3 each.

    For the locked n=24 this yields base=14, missing_function=5, missing_parameter=5.
    For other n it scales the same ~60/20/20 shape while keeping base the plurality and
    every sub_type present whenever n >= 3.
    """
    if n <= 0:
        return []
    if n < 3:
        return ["base"] * n
    mf = max(1, round(n * 0.21))
    mp = max(1, round(n * 0.21))
    base = n - mf - mp
    # Guarantee base remains the strict plurality.
    while base <= max(mf, mp):
        if mf >= mp and mf > 1:
            mf -= 1
        elif mp > 1:
            mp -= 1
        else:
            break
        base = n - mf - mp
    return ["base"] * base + ["missing_function"] * mf + ["missing_parameter"] * mp


def generate_multiturn_tasks(n: int = 24, seed: int = 42) -> list[dict]:
    """Deterministically build ``n`` multi-turn tool-use task rows (default pool of 24).

    For each task we pick a template (round-robin over the 8 registered templates) and a
    sub_type (from a fixed schedule), instantiate it with a single seeded ``random.Random``,
    then derive ``expected_states`` by REPLAYING the template's ``reference_calls`` turn by
    turn through a fresh ``ToolSandbox(initial_state)``. The returned rows are in the JSONL
    shape consumed by ``MultiTurnToolWorkload.instances()``:

        {"id", "turns", "initial_state", "expected_states", "sub_type"}

    plus a debug ``_reference_calls`` key (the replay sequence) so tests can prove the
    oracle is achievable. ``id`` is the stable ``mt_{i:02d}_{sub_type}``.
    """
    rng = random.Random(seed)
    subtypes = _subtype_schedule(n)
    rows: list[dict] = []
    for i in range(n):
        template = _TEMPLATES[i % len(_TEMPLATES)]
        sub_type = subtypes[i]
        task = template(rng, sub_type)
        initial_state = task["initial_state"]
        reference_calls: list[list[dict]] = task["reference_calls"]

        # Replay -> expected post-turn states (the oracle; never hand-written).
        sandbox = ToolSandbox(initial_state=initial_state)
        expected_states: list[dict] = []
        for calls in reference_calls:
            for call in calls:
                sandbox.dispatch(call)
            expected_states.append(sandbox.state())

        rows.append(
            {
                "id": f"mt_{i:02d}_{sub_type}",
                "turns": task["turns"],
                "initial_state": initial_state,
                "expected_states": expected_states,
                "sub_type": sub_type,
                "_reference_calls": reference_calls,
            }
        )
    return rows
