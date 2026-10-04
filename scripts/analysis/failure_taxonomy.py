"""Classify saved cell outputs into failure modes; report MAXN->throttled and
cross-tier shift. 0 compute (post-hoc over committed .traces.jsonl)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter

_REFUSAL = ("i cannot", "i can't", "i'm unable", "as an ai", "cannot help")


def classify(*, score: float, output: str, target: str) -> str:
    if score >= 1.0:
        return "correct"
    o = output.strip()
    if not o:
        return "empty"
    if any(p in o.lower() for p in _REFUSAL):
        return "refusal"
    if len(o.split()) > 500:
        return "runaway"
    return "wrong_answer"


def cell_counts(traces_path: Path) -> Counter:
    rows = [json.loads(l) for l in traces_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return Counter(classify(score=r["score"], output=r["output"], target=str(r.get("target", "")))
                   for r in rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)
    L = ["# Failure-mode taxonomy\n", "| dir | cell | correct | wrong | refusal | empty | runaway |", "|---|---|--:|--:|--:|--:|--:|"]
    for d in args.dirs:
        for tp in sorted(Path(d).glob("*.traces.jsonl")):
            c = cell_counts(tp)
            L.append(f"| {Path(d).name} | {tp.name.replace('.traces.jsonl','')} | "
                     f"{c['correct']} | {c['wrong_answer']} | {c['refusal']} | {c['empty']} | {c['runaway']} |")
    L.append("\n† tooluse rows are graded by the BFCL live_simple checker; all three models score ~0 (prose, not BFCL JSON) — a real but degenerate axis. Exclude tooluse from cross-workload failure-mode comparisons.")
    md = "\n".join(L) + "\n"
    if args.out != "-":
        Path(args.out).write_text(md, encoding="utf-8")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
