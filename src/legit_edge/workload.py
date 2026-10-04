"""Workload abstraction. A Workload provides instances and a score function."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import json
import re


CONFIDENCE_SUFFIX = (
    "\n\nAfter your answer, on a new final line output exactly "
    "'Confidence: N%' where N (0-100) is your confidence that your answer is correct."
)


def parse_confidence(text: str) -> float | None:
    m = re.findall(r"[Cc]onfidence:\s*([0-9]+(?:\.[0-9]+)?)\s*%?", text)
    if not m:
        return None
    val = float(m[-1]) / 100.0
    return max(0.0, min(1.0, val))


@dataclass
class Instance:
    input: str
    target: str
    raw: dict


class Workload(Protocol):
    name: str

    def instances(self) -> list[Instance]: ...
    def score(self, instance: Instance, output: str) -> float: ...


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class MathWorkload:
    jsonl_path: Path
    name: str = "math"

    def instances(self) -> list[Instance]:
        return [
            Instance(input=str(d["input"]), target=str(d["target"]), raw=d)
            for d in _load_jsonl(self.jsonl_path)
        ]

    def score(self, instance: Instance, output: str) -> float:
        # Extract the last number from output and compare to target as floats.
        # Float comparison avoids the rstrip(".0") bug that would corrupt "100" -> "1".
        nums = re.findall(r"-?\d+(?:\.\d+)?", output)
        if not nums:
            return 0.0
        try:
            return 1.0 if float(nums[-1]) == float(instance.target.strip()) else 0.0
        except ValueError:
            return 0.0


@dataclass
class ReasoningWorkload:
    jsonl_path: Path
    name: str = "reasoning"

    def instances(self) -> list[Instance]:
        return [
            Instance(
                input=str(d.get("input", d.get("question", ""))),
                target=str(d.get("target", d.get("answer", ""))),
                raw=d,
            )
            for d in _load_jsonl(self.jsonl_path)
        ]

    def score(self, instance: Instance, output: str) -> float:
        # BBH answers are typically letters or short strings; case-insensitive substring on tail
        return 1.0 if instance.target.strip().lower() in output.strip().lower()[-200:] else 0.0


@dataclass
class ToolUseWorkload:
    jsonl_path: Path
    ground_truth_path: Path | None = None
    name: str = "tooluse"

    def __post_init__(self):
        if self.ground_truth_path is None:
            self.ground_truth_path = self.jsonl_path.parent / "tooluse_ground_truth.jsonl"

    def _load_ground_truth(self) -> dict:
        if not self.ground_truth_path.exists():
            return {}
        return {row["id"]: row for row in _load_jsonl(self.ground_truth_path)}

    def instances(self) -> list[Instance]:
        gt = self._load_ground_truth()
        out = []
        for d in _load_jsonl(self.jsonl_path):
            qid = d.get("id", "")
            gt_row = gt.get(qid, {})
            out.append(Instance(
                input=json.dumps({"question": d.get("question"), "function": d.get("function", [])}),
                target=json.dumps({
                    "id": qid,
                    "function": d.get("function", []),
                    "ground_truth": gt_row.get("ground_truth", []),
                }),
                raw=d,
            ))
        return out

    def score(self, instance: Instance, output: str) -> float:
        from .bfcl_checker import simple_function_checker, Language
        target = json.loads(instance.target)
        if not target.get("ground_truth"):
            return 0.0
        try:
            model_calls = json.loads(output) if output.strip().startswith("[") else []
        except Exception:
            model_calls = []
        if not model_calls:
            return 0.0
        func_descs = target.get("function", [])
        gt_list = target["ground_truth"]
        # Score each model call against its corresponding ground-truth entry
        # (simple category: exactly one call expected)
        if len(model_calls) != 1 or len(func_descs) < 1 or len(gt_list) < 1:
            return 0.0
        try:
            result = simple_function_checker(
                func_descs[0], model_calls[0], gt_list[0], Language.PYTHON, "legit-edge",
            )
            return 1.0 if result.get("valid") else 0.0
        except Exception:
            return 0.0


@dataclass
class MultiTurnToolWorkload:
    """Bespoke multi-step tool-use workload (Path B; same step-level metrics as BFCL multi-turn).

    A task is a multi-turn (2–4 user turns) scenario with state-dependent turns. Grading is by
    REPLAY: a recorded trace's tool calls are dispatched (per turn) through a fresh
    ``ToolSandbox`` seeded from ``initial_state``; after each turn the sandbox ``state()`` must
    equal that turn's pre-registered expected post-turn state (state-diff oracle). The live
    per-turn agent loop that PRODUCES the trace lives in the runner — this class only scores.
    """

    jsonl_path: Path | None
    name: str = "tooluse_mt"

    def instances(self) -> list[Instance]:
        if self.jsonl_path is None:
            return []
        out = []
        for d in _load_jsonl(self.jsonl_path):
            task_id = d.get("id", "")
            turns = d.get("turns", [])
            initial_state = d.get("initial_state", {})
            expected_states = d.get("expected_states", [])
            out.append(Instance(
                input=json.dumps(
                    {"id": task_id, "turns": turns, "initial_state": initial_state}
                ),
                target=json.dumps({
                    "id": task_id,
                    "initial_state": initial_state,
                    "expected_states": expected_states,
                }),
                raw=d,
            ))
        return out

    def score_detailed(self, instance: Instance, trace) -> dict:
        """Grade a recorded multi-turn trace by replay against the expected per-turn states.

        ``trace`` is ``{"turns": [{"calls": [call, ...], "terminated": bool}, ...]}`` — a Python
        dict OR a JSON string (both accepted). Returns the locked contract keys
        ``{task_success, turn_failures, nonterminated}`` plus raw ``n_turns`` / ``failed_turns``
        so a cell-level analysis can compute the pre-registered micro-averaged
        turn_failure_rate = (sum failed turns) / (sum turns).
        """
        from .tool_sandbox import ToolSandbox

        if isinstance(trace, str):
            trace = json.loads(trace)
        target = json.loads(instance.target)
        initial_state = target.get("initial_state", {})
        expected_states = target.get("expected_states", [])
        trace_turns = trace.get("turns", []) if isinstance(trace, dict) else []

        sandbox = ToolSandbox(initial_state=initial_state)
        turn_ok: list[bool] = []
        nonterminated = False
        for i, expected in enumerate(expected_states):
            if i < len(trace_turns):
                turn = trace_turns[i]
                for call in turn.get("calls", []):
                    sandbox.dispatch(call)
                turn_ok.append(sandbox.state() == expected)
                # Non-termination is measured only over the task's graded turns; the live
                # runner produces exactly one trace turn per task turn, so any trace turns
                # beyond expected_states are ignored for both state-diff and termination.
                if not turn.get("terminated", True):
                    nonterminated = True
            else:
                # Model gave up before this turn: counts as a failed turn.
                turn_ok.append(False)

        n_turns = len(turn_ok)
        failed = sum(1 for ok in turn_ok if not ok)
        all_ok = bool(turn_ok) and all(turn_ok)
        return {
            "task_success": 1.0 if (all_ok and not nonterminated) else 0.0,
            "turn_failures": failed / max(1, n_turns),
            "nonterminated": bool(nonterminated),
            "n_turns": n_turns,
            "failed_turns": failed,
        }

    def score(self, instance: Instance, output: str) -> float:
        """Single-generation pipeline shim: parse a JSON-encoded trace, return task_success."""
        return self.score_detailed(instance, output)["task_success"]
