"""Tests for the confound-aware calibration helper: confidence-line stripping +
re-score recovers true correctness, so ECE is computed against the answer (not the
confidence %)."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str((Path(__file__).parent.parent / "scripts" / "analysis").resolve()))


def test_strip_confidence():
    from reshoot_calibration import strip_confidence
    assert strip_confidence("The answer is 42.\nConfidence: 90%") == "The answer is 42."
    assert strip_confidence("...42\nconfidence: 5%") == "...42"
    assert strip_confidence("no conf here") == "no conf here"
    assert strip_confidence("The answer is 8\nConfidence: 100") == "The answer is 8"


def test_cell_calibration_rescores_stripped(tmp_path):
    from reshoot_calibration import cell_calibration
    rows = [
        {"instance_index": 0, "input": "2+2?", "target": "4",
         "output": "The answer is 4\nConfidence: 90%", "score": 0.0, "confidence": 0.90},
        {"instance_index": 1, "input": "3+3?", "target": "6",
         "output": "The answer is 7\nConfidence: 80%", "score": 0.0, "confidence": 0.80},
    ]
    p = tmp_path / "m__spark__math__run0.traces.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    c = cell_calibration(tmp_path, "m__spark__math")
    assert c["n"] == 2
    # row0 stripped -> "...4" == target 4 (correct); row1 -> 7 != 6 (wrong)
    assert abs(c["corrected_acc"] - 0.5) < 1e-9
    # raw score was 0.0 for both (grader had read the confidence number) -> confound visible
    assert c["raw_acc"] == 0.0
    # ECE over [(0.9,1),(0.8,0)] is finite and positive
    assert c["ece"] == c["ece"] and c["ece"] >= 0.0


def test_cell_pairs_returns_stripped_rescored_pairs(tmp_path):
    from reshoot_calibration import cell_pairs
    rows = [
        {"input": "2+2?", "target": "4",
         "output": "The answer is 4\nConfidence: 90%", "score": 0.0, "confidence": 0.90},
        {"input": "3+3?", "target": "6",
         "output": "The answer is 7\nConfidence: 80%", "score": 0.0, "confidence": 0.80},
    ]
    p = tmp_path / "m__spark__math__run0.traces.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    pairs = cell_pairs(tmp_path, "m__spark__math")
    # confidence kept; correctness from the confidence-STRIPPED answer (4==4 ok, 7!=6 wrong)
    assert pairs == [(0.90, 1), (0.80, 0)]
