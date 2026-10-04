"""Energy per output token, by tier -- separates energy from accuracy in energy-per-correct.

energy_per_correct mixes energy with accuracy: a less accurate cell pays
more joules per *correct* answer even at identical per-token efficiency (this is why
phi-3.5-mini math shows the high 0.69 Spark/Jetson ratio -- lower Spark accuracy, not
worse Spark efficiency). energy_per_token = sum(joules_interval) / sum(output_tokens)
isolates the hardware/runtime cost of generating a token from whether the answer is
correct. Post-hoc over committed standard-MAXN traces (0 new compute)."""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path

# 5-model study (3 general-purpose + 2 agentic-tuned). Cells with no result file on disk
# are skipped (e.g. a model that has not been run yet), so this
# works on partial 3-model data as well as the full 5-model grid.
MODELS = ["llama_3_1_8b", "qwen_2_5_7b", "phi_3_5_mini", "hermes3_8b", "llama3_groq_tooluse_8b"]
WORKLOADS = ["math", "reasoning"]   # tool-use excluded (degenerate)


def cell_j_per_token(traces: Path) -> float:
    if not traces.exists():
        return float("nan")
    total_j = total_tok = 0.0
    for line in traces.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        j, ot = r.get("joules_interval"), r.get("output_tokens")
        if j is None or not ot:
            continue
        total_j += float(j)
        total_tok += float(ot)
    return total_j / total_tok if total_tok else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jetson", required=True, help="jetson standard-MAXN run dir")
    ap.add_argument("--spark", required=True, help="spark standard run dir")
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)

    L = [
        "# Energy per output token, by tier (standard MAXN; de-confounds energy from accuracy)\n",
        "_sum(joules_interval) / sum(output_tokens) over the standard MAXN run; tool-use "
        "excluded. Contrast with energy-per-correct, which divides by *correct* "
        "answers and so folds in accuracy._\n",
        "| model `.` workload | Jetson J/tok | Spark J/tok | ratio (Spark/Jetson) |",
        "|---|--:|--:|--:|",
    ]
    ratios = []
    for model in MODELS:
        for wl in WORKLOADS:
            jpath = Path(args.jetson) / f"{model}__jetson__{wl}.traces.jsonl"
            spath = Path(args.spark) / f"{model}__spark__{wl}.traces.jsonl"
            # Skip cells with no committed data on either tier (e.g. a model that has
            # not been run) so the table/summary stay clean.
            if not jpath.exists() and not spath.exists():
                continue
            jj = cell_j_per_token(jpath)
            sj = cell_j_per_token(spath)
            ratio = sj / jj if jj == jj and jj else float("nan")
            if ratio == ratio:
                ratios.append(ratio)
            L.append(f"| {model} `.` {wl} | {jj:.3f} | {sj:.3f} | {ratio:.2f} |")
    if ratios:
        L += [
            "",
            f"**Spark/Jetson J/token ratio: median {statistics.median(ratios):.2f}, "
            f"range {min(ratios):.2f}-{max(ratios):.2f}** (compare the energy-per-correct "
            f"ratio range 0.39-0.69, median 0.46, whose 0.69 outlier was an accuracy effect).",
        ]
    else:
        L += ["", "_No cells with committed energy traces found._"]
    md = "\n".join(L) + "\n"
    if args.out != "-":
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
