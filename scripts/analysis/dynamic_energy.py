"""Overhead-subtracted dynamic energy for the Jetson standard runs.

Decomposes each Jetson standard cell's energy into three measured quantities so the
"board overhead" share of the ~2x cross-tier energy gap can be bounded:

  whole-board (VDD_IN)        = sum(joules_interval)        per-instance integral of the
                                whole-board rail (power_W_mean_interval ~= VDD_IN, ~18-20 W
                                under LLM load; board idle ~6 W MAXN / ~4.2 W throttled).
  GPU-domain (VDD_CPU_GPU_CV) = energy_gpu_domain_j         summary integral of the narrow
                                compute rail (~8-9 W under load; idle ~1.57 W MAXN / 0.57 W
                                throttled). The Orin Nano INA3221 cannot isolate GPU alone;
                                VDD_CPU_GPU_CV is the combined CPU+GPU compute rail and the
                                honest analog to Spark's nvidia-smi GPU-domain power.draw.
  dynamic (compute)           = energy_gpu_domain_j - idle_W * duration
                                idle-subtracted dynamic energy on the compute rail.

duration is approximated by the wall-clock proxy sum(latency_s) (serial cell; one prompt
at a time, so the per-instance latencies tile the cell's active window). idle_W is the
GPU-domain idle baseline for the regime (MAXN 1.568 W, throttled 0.568 W; see
results/jetson-maxn-idle-baseline-20260607.txt + jetson-throttled-prerun-state-20260608.txt).

Board-overhead share = (whole-board - GPU-domain) / whole-board: the fraction of the
board's energy that is NOT on the compute rail (SoC / mem / IO / fixed platform draw).
A large share means the ~2x cross-tier energy-per-token gap is partly a fixed-platform
artifact of the Orin Nano dev board, not purely a compute-efficiency difference.

Post-hoc over committed Jetson standard traces; 0 new compute.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

# GPU-domain (VDD_CPU_GPU_CV) idle baselines, watts, per regime.
IDLE_W = {"MAXN": 1.568, "throttled": 0.568}
# Whole-board (VDD_IN) idle baselines, watts, per regime (context for the board floor).
IDLE_VDD_IN_W = {"MAXN": 6.000, "throttled": 4.216}

MODELS = ["llama_3_1_8b", "qwen_2_5_7b", "phi_3_5_mini", "hermes3_8b", "llama3_groq_tooluse_8b"]
WORKLOADS = ["math", "reasoning"]   # tool-use excluded (degenerate capability)


def cell_energy(run_dir: Path, cell: str, regime: str) -> dict | None:
    summ_path = run_dir / f"{cell}.json"
    traces_path = run_dir / f"{cell}.traces.jsonl"
    if not summ_path.exists() or not traces_path.exists():
        return None
    summ = json.loads(summ_path.read_text(encoding="utf-8"))
    gpu_domain_j = float(summ.get("energy_gpu_domain_j", float("nan")))
    board_j = 0.0
    duration_s = 0.0
    n = 0
    for line in traces_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ji = r.get("joules_interval")
        lat = r.get("latency_s")
        if ji is not None:
            board_j += float(ji)
        if lat is not None:
            duration_s += float(lat)
        n += 1
    idle_w = IDLE_W[regime]
    idle_energy_j = idle_w * duration_s
    dynamic_j = gpu_domain_j - idle_energy_j
    board_overhead_j = board_j - gpu_domain_j
    return {
        "cell": cell, "regime": regime, "n": n,
        "duration_s": duration_s,
        "board_j": board_j,                       # whole-board VDD_IN integral
        "gpu_domain_j": gpu_domain_j,             # VDD_CPU_GPU_CV integral
        "idle_energy_j": idle_energy_j,           # idle GPU-domain over the window
        "dynamic_j": dynamic_j,                   # idle-subtracted dynamic compute
        "board_overhead_j": board_overhead_j,     # board - compute rail
        "board_overhead_share": (board_overhead_j / board_j) if board_j else float("nan"),
        "domain_share_of_board": (gpu_domain_j / board_j) if board_j else float("nan"),
        "dynamic_share_of_domain": (dynamic_j / gpu_domain_j) if gpu_domain_j else float("nan"),
        "mean_board_W": (board_j / duration_s) if duration_s else float("nan"),
        "mean_domain_W": (gpu_domain_j / duration_s) if duration_s else float("nan"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxn", required=True, help="Jetson standard MAXN run dir")
    ap.add_argument("--throttled", required=True, help="Jetson standard throttled run dir")
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)

    rows = []
    for regime, rd in [("MAXN", Path(args.maxn)), ("throttled", Path(args.throttled))]:
        for model in MODELS:
            for wl in WORKLOADS:
                cell = f"{model}__jetson__{wl}"
                r = cell_energy(rd, cell, regime)
                if r is not None:
                    rows.append(r)

    L = [
        "# Overhead-subtracted dynamic energy — Jetson standard\n",
        "_Three measured energy quantities per cell. **whole-board** = sum(joules_interval) "
        "(VDD_IN rail, ~18-20 W under load); **GPU-domain** = energy_gpu_domain_j "
        "(VDD_CPU_GPU_CV compute rail, ~8-9 W under load); **dynamic** = GPU-domain − "
        "idle_W×duration (idle GPU-domain baseline: MAXN 1.568 W, throttled 0.568 W). "
        "duration ≈ sum(latency_s) (serial cell). **board-overhead share** = "
        "(whole-board − GPU-domain)/whole-board = fraction of board energy off the compute "
        "rail (fixed SoC/mem/IO/platform draw)._\n",
        "| cell | regime | n | duration (s) | whole-board J (VDD_IN) | GPU-domain J | "
        "idle J | dynamic J | board-overhead share | dynamic/domain |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        L.append(
            f"| {r['cell']} | {r['regime']} | {r['n']} | {r['duration_s']:.0f} | "
            f"{r['board_j']:.0f} | {r['gpu_domain_j']:.0f} | {r['idle_energy_j']:.0f} | "
            f"{r['dynamic_j']:.0f} | {r['board_overhead_share']:.3f} | "
            f"{r['dynamic_share_of_domain']:.3f} |"
        )

    # Aggregate board-overhead share by regime
    for regime in ["MAXN", "throttled"]:
        sub = [r for r in rows if r["regime"] == regime]
        if sub:
            shares = [r["board_overhead_share"] for r in sub]
            dom = [r["domain_share_of_board"] for r in sub]
            mean_share = sum(shares) / len(shares)
            mean_dom = sum(dom) / len(dom)
            L += [
                "",
                f"**{regime}: mean board-overhead share = {mean_share:.3f}** "
                f"(GPU-domain is {mean_dom:.1%} of whole-board energy, "
                f"range {min(shares):.3f}–{max(shares):.3f} over {len(sub)} cells).",
            ]

    L += [
        "",
        "## Interpretation",
        "- A cross-tier energy-per-token ratio computed from each tier's primary rail compares "
        "Spark GPU-domain (nvidia-smi) against Jetson **whole-board** VDD_IN; the matched "
        "compute-rail comparison is in energy_per_token_railcompare.py.",
        "- On Jetson, roughly **half** of the *whole-board* energy is fixed platform overhead "
        "(VDD_SOC + IO + the dev-board's static VDD_IN floor) that is NOT on the compute rail. "
        "If the cross-tier comparison were made at the **whole-board** level it would overstate "
        "the compute-efficiency gap, because the Orin Nano dev board carries a large fixed "
        "platform draw a deployed module would not.",
        "- The **dynamic** (idle-subtracted) compute energy is only modestly below the raw "
        "GPU-domain figure (idle floor is small relative to the ~8-9 W active compute draw), "
        "so subtracting idle does not materially change the per-correct/per-token energy ranking.",
    ]

    md = "\n".join(L) + "\n"
    if args.out != "-":
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out} ({len(rows)} cells)")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
