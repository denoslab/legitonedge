"""Run-to-run determinism over k repeated trials of the same cell.
D = mean over instances of (fraction of the k runs equal to the modal output);
score_flip_rate = fraction of instances whose 0/1 score is not constant."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter
import numpy as np


def _load(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def determinism_for_cell(run_dir: Path, cell: str) -> dict:
    files = sorted(Path(run_dir).glob(f"{cell}__run*.traces.jsonl"))
    runs = [_load(f) for f in files]
    k = len(runs)
    n = min(len(r) for r in runs) if runs else 0
    modal_match, flips = [], 0
    for i in range(n):
        outs = [runs[j][i]["output"] for j in range(k)]
        modal_n = Counter(outs).most_common(1)[0][1]
        modal_match.append(modal_n / k)
        scores = {runs[j][i]["score"] for j in range(k)}
        if len(scores) > 1:
            flips += 1
    return {"cell": cell, "k": k, "n": n,
            "D": float(np.mean(modal_match)) if modal_match else float("nan"),
            "score_flip_rate": flips / n if n else float("nan")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)
    d = Path(args.run_dir)
    cells = sorted({f.name.split("__run")[0] for f in d.glob("*__run*.traces.jsonl")})
    L = ["# Determinism (k repeated trials)\n", "| cell | k | n | D (modal-match) | score-flip rate |", "|---|--:|--:|--:|--:|"]
    for c in cells:
        r = determinism_for_cell(d, c)
        L.append(f"| {c} | {r['k']} | {r['n']} | {r['D']:.3f} | {r['score_flip_rate']:.3f} |")
    md = "\n".join(L) + "\n"
    if args.out != "-":
        Path(args.out).write_text(md, encoding="utf-8")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
