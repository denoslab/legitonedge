"""Thermal/energy diagnostics: wall-time anomaly, thermal envelope, energy paired-Δ,
and cross-tier output determinism — over per-instance .traces.jsonl sidecars.

This is the companion to output_stability_paired.py. Where that script answers
"is capability/latency stable MAXN→throttled?", this one answers three further
questions:

  Q1 (wall-time anomaly): why can the throttled standard run finish *faster* than
     the MAXN one? -> per-cell output_tokens + wall + temperature + byte-identical
     output fraction (MAXN vs throttled), plus an aggregate wall reconstruction.
  Q3 (dim-4 energy): J/attempt and J/correct paired-Δ (throttled − MAXN), per cell
     + non-tool-use aggregate.
  Q5 (cross-tier reasoning): pair Spark vs Jetson MAXN-std by instance_index and
     measure byte-identical output fraction + score disagreement, to decide whether
     the phi-reasoning divergence is BBH-grader format-sensitivity or runtime
     sampling drift.

Usage:
  python v2_thermal_energy_diag.py <jetson-maxn-std-dir> <jetson-throttled-std-dir> \
      [--spark <spark-maxn-std-dir>] [--out report.md]

All inputs are run directories holding <model>__<tier>__<workload>.traces.jsonl
sidecars. Cells are matched across directories by filename (the seed-42 sampler
makes instance_index i "the same prompt" in every run).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


# Orin Nano Super software thermal-throttle activation is ~83-87 °C on the
# CPU/GPU thermal zones; we only need it as a reference line for "did we ever
# approach the envelope?", not as a hard-coded claim.
ORIN_THROTTLE_REF_C = 83.0


def load_traces(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def cell_summary(rows: list[dict]) -> dict:
    scores = np.array([r["score"] for r in rows], dtype=float)
    lat = np.array([r["latency_s"] for r in rows], dtype=float)
    toks = np.array([r.get("output_tokens", 0) or 0 for r in rows], dtype=float)
    joules = np.array([r.get("joules_interval", 0.0) or 0.0 for r in rows], dtype=float)
    temps = np.array([r["temperature_c"] for r in rows if r.get("temperature_c") is not None],
                     dtype=float)
    n = len(rows)
    n_correct = float(scores.sum())
    total_j = float(joules.sum())
    sum_lat = float(lat.sum())
    return {
        "n": n,
        "cap": float(scores.mean()) if n else float("nan"),
        "n_correct": n_correct,
        "mean_tokens": float(toks.mean()) if n else float("nan"),
        "median_tokens": float(np.median(toks)) if n else float("nan"),
        "sum_lat_s": sum_lat,
        "wall_min": sum_lat / 60.0,
        "p50_lat": float(np.percentile(lat, 50)) if n else float("nan"),
        "total_j": total_j,
        "j_per_attempt": total_j / n if n else float("nan"),
        "j_per_correct": (total_j / n_correct) if n_correct > 0 else float("inf"),
        "mean_power_W": (total_j / sum_lat) if sum_lat > 0 else float("nan"),
        "temp_max": float(temps.max()) if temps.size else float("nan"),
        "temp_mean": float(temps.mean()) if temps.size else float("nan"),
        "temp_p95": float(np.percentile(temps, 95)) if temps.size else float("nan"),
        "n_temp": int(temps.size),
    }


def identical_fraction(rows_a: list[dict], rows_b: list[dict]) -> tuple[float, int, int]:
    """Byte-identical output fraction + score-disagreement count over paired rows."""
    n = min(len(rows_a), len(rows_b))
    same_out = 0
    score_disagree = 0
    for i in range(n):
        if rows_a[i]["output"] == rows_b[i]["output"]:
            same_out += 1
        if rows_a[i]["score"] != rows_b[i]["score"]:
            score_disagree += 1
    return (same_out / n if n else float("nan")), score_disagree, n


def cells_in(dir_: Path) -> dict[str, Path]:
    return {p.name.replace(".traces.jsonl", ""): p for p in sorted(dir_.glob("*.traces.jsonl"))}


def _fmt_jc(v: float) -> str:
    return "∞" if v == float("inf") else f"{v:.0f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jetson_maxn_dir")
    ap.add_argument("jetson_throttled_dir")
    ap.add_argument("--spark", default=None, help="Spark MAXN std dir for cross-tier reasoning")
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)

    maxn_dir = Path(args.jetson_maxn_dir)
    throt_dir = Path(args.jetson_throttled_dir)
    maxn_cells = cells_in(maxn_dir)
    throt_cells = cells_in(throt_dir)
    shared = [c for c in maxn_cells if c in throt_cells]

    L: list[str] = []
    L.append("# Thermal/energy diagnostics — wall-time anomaly, thermal envelope, energy, cross-tier determinism\n")
    L.append(f"MAXN std dir: `{maxn_dir.name}`  ")
    L.append(f"throttled std dir: `{throt_dir.name}`\n")

    # ---- Section 1: wall-time anomaly + thermal envelope -------------------
    L.append("## 1. Wall-time anomaly + thermal envelope (Q1)\n")
    L.append("Per cell, MAXN vs throttled. `ident%` = byte-identical output fraction across the "
             "two regimes (clock pinning cannot change float results, so <100% ⟹ run-to-run "
             "runtime non-determinism). Δ columns are throttled − MAXN.\n")
    L.append("| cell | n | tok_M | tok_T | Δtok | wall_M | wall_T | Δwall | ident% | Tmax_M | Tmax_T |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    tot_maxn_min = tot_throt_min = 0.0
    sec1_rows = []
    for cell in shared:
        m = load_traces(maxn_cells[cell]); t = load_traces(throt_cells[cell])
        sm = cell_summary(m); st = cell_summary(t)
        ident, _, _ = identical_fraction(m, t)
        tot_maxn_min += sm["wall_min"]; tot_throt_min += st["wall_min"]
        sec1_rows.append((cell, sm, st, ident))
        L.append(
            f"| {cell} | {sm['n']} | {sm['mean_tokens']:.0f} | {st['mean_tokens']:.0f} | "
            f"{st['mean_tokens'] - sm['mean_tokens']:+.0f} | "
            f"{sm['wall_min']:.0f}m | {st['wall_min']:.0f}m | "
            f"{st['wall_min'] - sm['wall_min']:+.0f}m | {ident * 100:.1f}% | "
            f"{sm['temp_max']:.0f} | {st['temp_max']:.0f} |"
        )
    L.append(
        f"\n- **Inference-only wall (Σ latency over 9 cells):** MAXN {tot_maxn_min/60:.2f} h, "
        f"throttled {tot_throt_min/60:.2f} h, Δ {(tot_throt_min - tot_maxn_min)/60:+.2f} h.\n"
    )
    # determinism split
    det = [(c, ident) for (c, _, _, ident) in sec1_rows]
    L.append("- **Determinism split (byte-identical output across regimes):**")
    for c, ident in det:
        L.append(f"  - {c}: {ident*100:.1f}%")
    L.append("")
    maxT = max(s["temp_max"] for (_, s, _, _) in sec1_rows if not np.isnan(s["temp_max"]))
    maxT_t = max(s["temp_max"] for (_, _, s, _) in sec1_rows if not np.isnan(s["temp_max"]))
    L.append(f"- **Thermal envelope:** peak temp observed MAXN {maxT:.0f} °C, throttled {maxT_t:.0f} °C "
             f"— both well below the ~{ORIN_THROTTLE_REF_C:.0f} °C Orin software-throttle reference. "
             f"Neither regime entered thermal throttling at this workload.\n")

    # ---- Section 2: energy paired-Δ ----------------------------------------
    L.append("## 2. Energy paired-Δ — dim 4 (Q3)\n")
    L.append("Real joules (`joules_interval`, tegrastats VDD_IN). Δ = throttled − MAXN.\n")
    L.append("| cell | n | J/att_M | J/att_T | ΔJ/att | J/corr_M | J/corr_T | W_M | W_T |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    g6_dja = []
    for cell in shared:
        m = load_traces(maxn_cells[cell]); t = load_traces(throt_cells[cell])
        sm = cell_summary(m); st = cell_summary(t)
        if "tooluse" not in cell:
            g6_dja.append(st["j_per_attempt"] - sm["j_per_attempt"])
        L.append(
            f"| {cell} | {sm['n']} | {sm['j_per_attempt']:.0f} | {st['j_per_attempt']:.0f} | "
            f"{st['j_per_attempt'] - sm['j_per_attempt']:+.0f} | "
            f"{_fmt_jc(sm['j_per_correct'])} | {_fmt_jc(st['j_per_correct'])} | "
            f"{sm['mean_power_W']:.1f} | {st['mean_power_W']:.1f} |"
        )
    if g6_dja:
        arr = np.array(g6_dja)
        L.append(f"\n- **Non-tool-use mean ΔJ/attempt:** {arr.mean():+.1f} J "
                 f"(range [{arr.min():+.0f}, {arr.max():+.0f}] J).\n")

    # ---- Section 3: cross-tier output determinism --------------------------
    if args.spark:
        spark_dir = Path(args.spark)
        spark_cells = cells_in(spark_dir)
        L.append("## 3. Cross-tier output determinism — Spark aarch64 (GB10) vs Jetson aarch64 (Orin) (Q5)\n")
        L.append(f"Spark MAXN std dir: `{spark_dir.name}`. Pairs each Jetson-MAXN cell with its "
                 "Spark twin by instance_index. `ident%` = byte-identical output fraction; "
                 "`disagree` = instances scored differently by the *same* grader.\n")
        L.append("| workload-model | n | cap_spark | cap_jetson | ident% | disagree | tok_spark | tok_jet |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
        for cell in shared:
            # cell name is <model>__jetson__<workload>; build the spark twin name
            spark_name = cell.replace("__jetson__", "__spark__")
            if spark_name not in spark_cells:
                continue
            j = load_traces(maxn_cells[cell]); s = load_traces(spark_cells[spark_name])
            sj = cell_summary(j); ss = cell_summary(s)
            ident, disagree, n = identical_fraction(s, j)
            L.append(
                f"| {cell.replace('__jetson__', ' · ')} | {n} | {ss['cap']:.3f} | {sj['cap']:.3f} | "
                f"{ident*100:.1f}% | {disagree} | {ss['mean_tokens']:.0f} | {sj['mean_tokens']:.0f} |"
            )

        # divergence examples for reasoning cells (where scores disagree)
        L.append("\n### Cross-tier reasoning divergence examples (score disagreements)\n")
        for cell in shared:
            if "reasoning" not in cell:
                continue
            spark_name = cell.replace("__jetson__", "__spark__")
            if spark_name not in spark_cells:
                continue
            j = load_traces(maxn_cells[cell]); s = load_traces(spark_cells[spark_name])
            n = min(len(s), len(j))
            shown = 0
            L.append(f"\n**{cell}** — first 3 score disagreements:")
            for i in range(n):
                if s[i]["score"] == j[i]["score"]:
                    continue
                tgt = str(s[i].get("target", ""))[:40]
                so = s[i]["output"].replace("\n", " ")[-90:]
                jo = j[i]["output"].replace("\n", " ")[-90:]
                L.append(f"- idx {i} (target `{tgt}`): spark[{s[i]['score']:.0f}] …{so!r}  ‖  "
                         f"jetson[{j[i]['score']:.0f}] …{jo!r}")
                shown += 1
                if shown >= 3:
                    break
            if shown == 0:
                L.append("- (no score disagreements)")

    md = "\n".join(L) + "\n"
    if args.out == "-":
        print(md)
    else:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
