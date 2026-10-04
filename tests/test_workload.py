from legit_edge.workload import (
    Workload,
    MathWorkload,
    ReasoningWorkload,
    ToolUseWorkload,
    Instance,
)


def test_math_workload_loads_subset(tmp_path):
    p = tmp_path / "math_megaquick.jsonl"
    p.write_text('{"input": "1+1?", "target": "2"}\n{"input": "2*3?", "target": "6"}\n')
    w = MathWorkload(jsonl_path=p)
    items = w.instances()
    assert len(items) == 2
    assert items[0].input == "1+1?"
    assert items[0].target == "2"


def test_math_workload_scores_correctly(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"input": "x", "target": "42"}\n')
    w = MathWorkload(jsonl_path=p)
    inst = w.instances()[0]
    assert w.score(inst, "The answer is 42.") == 1.0
    assert w.score(inst, "I think it is 41.") == 0.0


def test_math_workload_distinguishes_trailing_zero_targets(tmp_path):
    # Regression: rstrip(".0") would have made target=100 score 1.0 against output "10".
    p = tmp_path / "x.jsonl"
    p.write_text('{"input": "x", "target": "100"}\n')
    w = MathWorkload(jsonl_path=p)
    inst = w.instances()[0]
    assert w.score(inst, "The answer is 100.") == 1.0
    assert w.score(inst, "The answer is 10.") == 0.0


def test_tooluse_workload_uses_bfcl_grader(tmp_path):
    """Verify ToolUseWorkload.score() uses the BFCL AST grader."""
    questions = tmp_path / "tooluse_megaquick.jsonl"
    gt = tmp_path / "tooluse_ground_truth.jsonl"
    questions.write_text(
        '{"id":"live_simple_0","question":[[{"role":"user","content":"add 2 and 3"}]],'
        '"function":[{"name":"add","description":"Adds two integers.","parameters":{"type":"dict",'
        '"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}]}\n'
    )
    gt.write_text('{"id":"live_simple_0","ground_truth":[{"add":{"a":[2],"b":[3]}}]}\n')

    w = ToolUseWorkload(jsonl_path=questions, ground_truth_path=gt)
    items = w.instances()
    assert len(items) == 1

    correct = '[{"add": {"a": 2, "b": 3}}]'
    wrong = '[{"add": {"a": 99, "b": 3}}]'
    assert w.score(items[0], correct) == 1.0, "Correct call should score 1.0"
    assert w.score(items[0], wrong) == 0.0, "Wrong arg value should score 0.0"


def test_tooluse_workload_graceful_missing_gt(tmp_path):
    """Items with no ground-truth entry score 0.0."""
    questions = tmp_path / "tooluse_megaquick.jsonl"
    gt = tmp_path / "tooluse_ground_truth.jsonl"
    questions.write_text(
        '{"id":"live_multiple_999","question":[[{"role":"user","content":"do something"}]],'
        '"function":[{"name":"foo","description":"x","parameters":{"type":"dict",'
        '"properties":{},"required":[]}}]}\n'
    )
    gt.write_text("")  # empty GT file

    w = ToolUseWorkload(jsonl_path=questions, ground_truth_path=gt)
    items = w.instances()
    assert len(items) == 1
    # No GT available → score must be 0.0 regardless of model output
    assert w.score(items[0], '[{"foo": {}}]') == 0.0
