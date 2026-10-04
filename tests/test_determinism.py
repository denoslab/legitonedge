import json, sys
from pathlib import Path
sys.path.insert(0, str((Path(__file__).parent.parent / "scripts" / "analysis").resolve()))


def _trace(p, outputs, scores):
    rows = [{"instance_index": i, "output": o, "score": s} for i, (o, s) in enumerate(zip(outputs, scores))]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_determinism_modal_and_flip(tmp_path):
    from determinism import determinism_for_cell
    _trace(tmp_path / "c__t__math__run0.traces.jsonl", ["A", "B"], [1.0, 1.0])
    _trace(tmp_path / "c__t__math__run1.traces.jsonl", ["A", "B"], [1.0, 1.0])
    _trace(tmp_path / "c__t__math__run2.traces.jsonl", ["A", "C"], [1.0, 0.0])
    res = determinism_for_cell(tmp_path, "c__t__math")
    assert res["k"] == 3
    assert abs(res["D"] - ((1.0 + 2/3) / 2)) < 1e-9
    assert abs(res["score_flip_rate"] - 0.5) < 1e-9
