"""Corrected verbalized-confidence calibration (ECE) for the confidence-elicitation runs
(`legit-edge run ... --confidence --repeat 5`).

The confidence elicitation appends a 'Confidence: N%' line to each output.
Tail-based graders would then grade the confidence value, not the answer:
MathWorkload extracts the LAST number (which becomes the confidence %), and
ReasoningWorkload matches the output tail. So we RE-SCORE each output with the
confidence line stripped to recover TRUE correctness, then compute ECE over
(parsed_confidence, true_correctness), pooled across the k repeats per cell.

We report raw (confounded) vs corrected accuracy so the confound is explicit.
This is a post-hoc analysis over committed .traces.jsonl (0 compute).
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np
from legit_edge.metrics import CalibrationMetric
from legit_edge.workload import (
    MathWorkload, ReasoningWorkload, ToolUseWorkload, Instance, parse_confidence,
)

_WL = {"math": MathWorkload, "reasoning": ReasoningWorkload, "tooluse": ToolUseWorkload}
_CONF_TAIL = re.compile(r"\s*confidence\s*:\s*[0-9]+(?:\.[0-9]+)?\s*%?\s*$", re.IGNORECASE)


def strip_confidence(output: str) -> str:
    """Remove a trailing 'Confidence: N%' line so the grader sees only the answer."""
    return _CONF_TAIL.sub("", output.rstrip()).rstrip()


def _workload(wname: str):
    return _WL[wname](jsonl_path=Path("."))   # .score() does not read jsonl_path


def _cell_files(run_dir: Path, cell_base: str) -> list[Path]:
    """Trace files for a cell: the k>1 __run{j} shards, else the single-run file."""
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob(f"{cell_base}__run*.traces.jsonl"))
    if not files:
        single = run_dir / f"{cell_base}.traces.jsonl"
        files = [single] if single.exists() else []
    return files


def _cell_records(run_dir: Path, cell_base: str) -> list[tuple[float, int, float]]:
    """(confidence, true_correct, raw_score) per scored instance, pooled across k repeats.

    Each output is confidence-stripped before re-scoring so the grader reads the
    answer, not the appended 'Confidence: N%'. raw_score is the original (confounded)
    grade for contrast.
    """
    wl = _workload(cell_base.split("__")[2])
    recs: list[tuple[float, int, float]] = []
    for f in _cell_files(run_dir, cell_base):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            conf = r.get("confidence")
            if conf is None:
                conf = parse_confidence(r.get("output", ""))
            if conf is None:
                continue
            inst = Instance(input=r.get("input", ""), target=r.get("target", ""), raw={})
            true_score = wl.score(inst, strip_confidence(r["output"]))
            recs.append((float(conf), int(round(true_score)), float(r.get("score", 0.0))))
    return recs


def cell_pairs(run_dir: Path, cell_base: str) -> list[tuple[float, int]]:
    """(confidence, true_correct) pairs for a cell — the resampling unit for the ECE bootstrap."""
    return [(c, y) for c, y, _ in _cell_records(run_dir, cell_base)]


def cell_calibration(run_dir: Path, cell_base: str) -> dict:
    """Pooled ECE for one cell across its k repeats, scoring the confidence-stripped output."""
    recs = _cell_records(run_dir, cell_base)
    if not recs:
        nan = float("nan")
        return {"cell": cell_base, "n": 0, "ece": nan, "raw_acc": nan,
                "corrected_acc": nan, "mean_conf": nan}
    pairs = [(c, y) for c, y, _ in recs]
    ece = CalibrationMetric().compute(pairs).value["ece"]
    return {
        "cell": cell_base, "n": len(pairs), "ece": float(ece),
        "raw_acc": float(np.mean([raw for _, _, raw in recs])),
        "corrected_acc": float(np.mean([y for _, y, _ in recs])),
        "mean_conf": float(np.mean([c for c, _, _ in recs])),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)
    d = Path(args.run_dir)
    bases = sorted({f.name.split("__run")[0] for f in d.glob("*__run*.traces.jsonl")})
    if not bases:
        bases = sorted({f.name.replace(".traces.jsonl", "") for f in d.glob("*.traces.jsonl")})
    L = ["# Verbalized-confidence calibration (ECE; confidence-stripped re-score)\n",
         "_Models emit `Confidence: N%`; tail-based graders are re-run on the "
         "confidence-stripped answer to get true correctness. raw_acc shows the "
         "confounded score (grader read the confidence %) for contrast._\n",
         "| cell | n | mean_conf | corrected_acc | raw_acc (confounded) | ECE |",
         "|---|--:|--:|--:|--:|--:|"]
    for b in bases:
        c = cell_calibration(d, b)
        L.append(
            f"| {c['cell']} | {c['n']} | {c['mean_conf']:.3f} | "
            f"{c['corrected_acc']:.3f} | {c['raw_acc']:.3f} | {c['ece']:.3f} |"
        )
    md = "\n".join(L) + "\n"
    if args.out != "-":
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
