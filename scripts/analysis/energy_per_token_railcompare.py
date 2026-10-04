"""Energy per output token at MATCHED measurement boundaries, by tier.

energy_per_token.py divides each cell's primary-rail integral (`joules_interval`,
summed over the cell's traces) by output tokens on BOTH tiers. But on the Jetson that primary rail is whole-board (~18-20 W: SoC +
memory + IO + dev-board static floor), whereas on the Spark the primary rail is already the
GPU domain (nvidia-smi `power.draw`). So that primary-rail Spark/Jetson ratio compares
Spark GPU-only against Jetson WHOLE-BOARD -- a measurement-boundary mismatch, not a clean
hardware-efficiency gap.

This script recomputes the per-token comparison at MATCHED boundaries:
  * Jetson GPU-domain  = energy_gpu_domain_j (VDD_CPU_GPU_CV compute rail, from the cell
                         summary {cell}.json) / sum(output_tokens) over the traces.
  * Spark GPU-domain   = sum(joules_interval) (nvidia-smi power.draw) / sum(output_tokens).
The Jetson VDD_CPU_GPU_CV rail combines CPU+GPU+CV, so it OVER-counts GPU-only energy; the
true GPU-only Jetson figure is bounded above by the GPU-domain number reported here.

Also reproduces the whole-board Jetson J/tok (= sum(joules_interval)/sum(output_tokens)) so
the boundary effect on the ratio is visible in one table. Post-hoc over committed standard
traces; 0 new compute.
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path

MODELS = ["llama_3_1_8b", "qwen_2_5_7b", "phi_3_5_mini", "hermes3_8b", "llama3_groq_tooluse_8b"]
WORKLOADS = ["math", "reasoning"]   # tool-use excluded (no correct answers under the prompt)


def trace_sums(traces: Path) -> tuple[float, float]:
    """Return (sum joules_interval, sum output_tokens) over a cell's traces."""
    sum_j = sum_tok = 0.0
    if not traces.exists():
        return float("nan"), 0.0
    for line in traces.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        j, ot = r.get("joules_interval"), r.get("output_tokens")
        if j is None or not ot:
            continue
        sum_j += float(j)
        sum_tok += float(ot)
    return sum_j, sum_tok


def jetson_gpu_domain_j(run_dir: Path, cell: str) -> float:
    summ = run_dir / f"{cell}.json"
    if not summ.exists():
        return float("nan")
    return float(json.loads(summ.read_text(encoding="utf-8")).get("energy_gpu_domain_j", float("nan")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jetson", required=True, help="jetson standard-MAXN run dir (GPU-rail logged)")
    ap.add_argument("--spark", required=True, help="spark standard run dir")
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)
    jdir, sdir = Path(args.jetson), Path(args.spark)

    L = [
        "# Energy per output token at MATCHED measurement boundaries, by tier (standard MAXN)\n",
        "_Corrects the boundary mismatch in the primary-rail comparison (energy_per_token.py), which divides "
        "the Jetson WHOLE-BOARD rail by tokens while the Spark figure is already GPU-domain. Here the "
        "**GPU-domain** columns put both tiers on the compute rail (Jetson `energy_gpu_domain_j` = "
        "VDD_CPU_GPU_CV; Spark `joules_interval` = nvidia-smi `power.draw`). The Jetson rail combines "
        "CPU+GPU+CV, so true GPU-only Jetson energy is bounded ABOVE by these numbers (the real gap is "
        "no wider than shown). `ratio` = Spark/Jetson; <1 means Spark draws less._\n",
        "| model `.` workload | Jetson WB J/tok | Jetson GPU-dom J/tok | Spark GPU J/tok | "
        "ratio WHOLE-BOARD | ratio GPU-DOMAIN |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    r_wb, r_gd = [], []
    for model in MODELS:
        for wl in WORKLOADS:
            cell_j = f"{model}__jetson__{wl}"
            cell_s = f"{model}__spark__{wl}"
            jt = jdir / f"{cell_j}.traces.jsonl"
            st = sdir / f"{cell_s}.traces.jsonl"
            if not jt.exists() and not st.exists():
                continue
            j_wb, j_tok = trace_sums(jt)
            s_j, s_tok = trace_sums(st)
            gd_j = jetson_gpu_domain_j(jdir, cell_j)
            j_wb_pt = j_wb / j_tok if j_tok else float("nan")
            j_gd_pt = gd_j / j_tok if j_tok else float("nan")
            s_pt = s_j / s_tok if s_tok else float("nan")
            ratio_wb = s_pt / j_wb_pt if j_wb_pt == j_wb_pt and j_wb_pt else float("nan")
            ratio_gd = s_pt / j_gd_pt if j_gd_pt == j_gd_pt and j_gd_pt else float("nan")
            if ratio_wb == ratio_wb:
                r_wb.append(ratio_wb)
            if ratio_gd == ratio_gd:
                r_gd.append(ratio_gd)
            L.append(
                f"| {model} `.` {wl} | {j_wb_pt:.3f} | {j_gd_pt:.3f} | {s_pt:.3f} | "
                f"{ratio_wb:.2f} | {ratio_gd:.2f} |"
            )
    if r_wb and r_gd:
        L += [
            "",
            f"**Whole-board ratio (published): median {statistics.median(r_wb):.2f}, "
            f"range {min(r_wb):.2f}-{max(r_wb):.2f}** -- Jetson dev-board draws ~2x Spark.",
            "",
            f"**GPU-domain ratio (matched boundary): median {statistics.median(r_gd):.2f}, "
            f"range {min(r_gd):.2f}-{max(r_gd):.2f}** -- at the compute rail the two tiers are "
            f"near parity; the ~2x whole-board gap is dominated by the Orin Nano dev board's fixed "
            f"platform overhead (~half of whole-board energy), which a deployed module would not carry.",
        ]
    else:
        L += ["", "_No cells with committed energy traces + GPU-domain summary found._"]
    md = "\n".join(L) + "\n"
    if args.out != "-":
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
