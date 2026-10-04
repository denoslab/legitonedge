# LegitOnEdge — Pre-Registered Protocol

**Date committed:** 2026-05-11 (in the original private repository)
**Note:** This is the original protocol and its dated amendments, retained for the record with only
internal cross-references (commit hashes, design-document sections) removed. The base protocol
predates the expansion from three to five models, which is declared in amendment E.5 below.

## Hypotheses

- **H1 (Output Stability × Calibration interaction):** ECE measured at MAXN over-predicts calibration accuracy at the throttled regime by ≥5 percentage points across the 18 cells. Two-sided test, α=0.10 after Holm-Bonferroni correction across 18 cells.
- **H2 (Mega-quick / Standard agreement):** median extrapolation error |Mega-quick estimate − Standard estimate| / Standard estimate ≤ 15% across the 5 reliability dimensions, computed at the per-cell level. Pre-registered tolerance band 15%.
- **H3 (Capability-reliability orthogonality):** Pearson correlation between composite capability score and composite LegitOnEdge reliability score across the 18 cells is below 0.5 (weakly correlated). Two-sided test on Fisher-z-transformed r.

## Cells

3 models (Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, Phi-3.5-mini-instruct) × 2 hardware tiers (Jetson Orin Nano 8GB / DGX Spark) × 3 workloads (math / reasoning / tool use) = **18 cells**. Standard mode doubles to 36 measurement units across two thermal regimes (MAXN, throttled).

**Runtime** (revised 2026-05-12 from the original 2026-05-11 wording): both tiers serve via Ollama (`OLLAMA_HOST=0.0.0.0 ollama serve &`) at quantization Q4_K_M by default. The "tier" dimension is hardware capacity, not runtime or quantization. This is a pre-experiment correction: no measurements had been collected against the prior wording (Jetson Q4_K_M Ollama + Spark FP16 vLLM). The 18-cell layout is unchanged in shape; only the *content* of the tier dimension is corrected. Hypotheses H1, H2, H3 are unchanged.

## Datasets (pinned with seed=42 via `legit-edge pin datasets`)

| Workload | Source | Standard set | Mega-quick set |
|---|---|---:|---:|
| math_gsm8k_hard | `reasoning-machines/gsm-hard` (train split) | 200 | 17 |
| reasoning_bbh | `lukaemon/bbh` pooled across 3 BBH-Hard subtasks (logical_deduction_seven_objects, tracking_shuffled_objects_seven_objects, navigate) | 201 | 17 |
| tooluse_bfcl | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` v3 `live_simple` only (258 items; GT from BFCL v4 `live_simple` possible-answer file — ID-stable, 100% overlap verified) | 100 | 16 |

The BFCL tooluse pool is **`live_simple` only** (258 items). This is a pre-experiment correction: the original protocol listed a 4-category mix (~1351 items, `live_multiple`-dominated ~78%), but the BFCL grader only has GT for `live_simple`; keeping the mixed pool would leave ~85% of standard rows and 100% of mega-quick rows scoring 0.0 by missing-GT fallback. Narrowing to `live_simple` gives 100% GT coverage and a real capability measurement. Ground truth is sourced from the BFCL v4 `live_simple` possible-answer file (cross-version is acceptable — IDs are stable for `live_simple`, overlap verified). Sampled subsets (n=16 mq, n=100 std, seed=42) and all other workload parameters are unchanged. `live_relevance` and `live_irrelevance` remain excluded (binary classification, distinct scoring methodology). All samples are deterministic given the pinned configuration and seed.

## Metrics

5 reliability dimensions per cell — (1) latency tail (p95, p99), (2) throughput stability, (3) output stability under thermal stress, (4) energy-per-correct-output, (5) calibration reliability (ECE) — plus capability accuracy per cell, plus the LegitOnEdge composite score per cell extending islegit.ai's formula `r·(1−d)·Σ(wᵢ·cᵢ)` with an explicit reliability factor R.

## Statistical methodology

- Bootstrap 95% CIs (B=10,000) on every per-cell scalar
- Block-bootstrap (block size = 10) for tail-latency metrics
- Holm-Bonferroni multiple-testing correction across 18 cells when reporting per-cell significance
- Deterministic seed = 42 throughout (`random`, `numpy`, `torch`)

## Exploratory analyses

Anything not pre-registered above will be reported in a clearly labelled "Exploratory analyses (post-hoc)" section.

### Amendment 2026-05-30 [pre-exp E.4] — repeat-run determinism and confidence-elicitation metrics & bounds (declared before any such data)
- **R1 determinism metric:** k=5 repeated trials at temperature=0, seed=42. Per cell:
  D = mean over instances of (fraction of the k runs equal to the per-instance modal
  output); score_flip_rate = fraction of instances whose 0/1 score is not constant;
  first_divergence_index = mean token index of the first cross-run difference. Bootstrap
  95% CI (B=10,000) on D over instances.
- **R2 thermal-stress workload + pilot gate:** stress variant = num_predict>=1024, zero
  inter-prompt idle, plus a >=20 min soak preamble. PILOT GO iff a 15-min Jetson heavy-load
  loop reaches sustained tegrastats peak >= 82 C; else NO-GO (recorded as a finding) and
  dim-3 leans on R1.
- **R3 verbalized confidence:** answer format gains a final line `Confidence: NN%` (0-100).
  ECE = 10 equal-width bins over (stated_confidence/100, correctness). Reported per cell,
  both tiers, both regimes. Weaker than logprob-ECE (disclosed); symmetric across tiers.
- **F-A composite normalization (pre-declared bounds):** latency_score = clip(1 - p99/(2*60s)),
  energy_score = clip(1 - Jpc/(2*2000J)), throughput_score penalizes only negative slope vs
  ref 0.05 tps/min. reliability_composite = weighted geometric mean of the 5 normalized dim
  scores (capability EXCLUDED — it is the scatter X-axis). H3 sensitivity spans: bounds +/-50%,
  bound-basis {absolute-SLA, per-tier-relative}, and weights {uniform, each named profile}.

### Amendment 2026-06-05 [pre-exp E.5] — expansion: agentic models + multi-step tool workload (declared before any expansion data)

**Timing discipline.** Declared and committed BEFORE any expansion (5-model / multi-step) measurement
is observed. Every bound / τ / threshold below is carried over UNCHANGED from the original protocol
and amendment E.4 and was NOT re-chosen with knowledge of any expansion-phase distribution (limiting
researcher degrees of freedom).

- **E.5.1 New models (→ 5 total).** Add two open-weight, function-calling-tuned models, served as
  identical Ollama tags on both tiers at Q4_K_M, 8 GB-fit verified (both 8.0B llama-arch, `tools`
  capability, 4.9 GB):
  - Hermes 3 (Llama-3.1-8B): `hermes3:8b-llama3.1-q4_K_M`
  - Llama-3-Groq-Tool-Use 8B: `llama3-groq-tool-use:8b-q4_K_M`

- **E.5.2 Cells (→ 30 base).** Base grid expands to 5 models × 2 tiers × 3 capability workloads
  (math, reasoning, tool-use-single) = **30 base cells** (was 18); Standard mode × 2 thermal regimes
  (MAXN, throttled) unchanged. The two new models run the COMPLETE core grid the original three run
  (full parity, incl. the Jetson naturally-throttled standard runs) → rectangular dataset.
  Determinism (k=5, T=0; amendment E.4 R1) and verbalized-confidence ECE (k=5; E.4 R3) extend to the
  two new models at mega-quick scale.

- **E.5.3 New workload — multi-step tool use (the agentic axis).** A controlled, on-device,
  multi-step tool-use workload, run across 5 models × 2 tiers × modes/regimes as a 4th workload.
  *Grader basis:* the official BFCL multi-turn checker is NOT used — it requires a stateful execution
  backend, two of its splits need network / heavy ML deps, the `bfcl-eval` package is not installable
  on aarch64/JetPack, and it `eval()`s model output; infeasible self-contained on the Orin (feasibility
  check, 2026-06-05). We use a bespoke equivalent with the same step-level metric outputs:
  - *Environment:* a deterministic, stdlib-only stateful tool sandbox with a fixed, whitelisted
    toolset over two domains — a notes/key-value store (`set_note, get_note, list_notes, delete_note`)
    and a numeric ledger (`ledger_add, ledger_subtract, ledger_balance, ledger_reset`) — plus a few
    decoy/irrelevant tools to probe missing-function handling. Model tool calls are parsed as a JSON
    list (as in `live_simple`) and **dispatched by name to Python callables (NO eval/exec)**.
  - *Tasks:* multi-turn scenarios (2–4 user turns) with **state-dependent turns** (turn N depends on
    state established by turn N−1). Authored as ~8 parameterized templates, instantiated with seed=42
    to a pinned pool; **standard = 24 tasks, mega-quick = 8 tasks** (fixed seed-42 subset). Sizes are
    smaller than the single-turn workloads (100/16) because each multi-turn task is multiple
    generations; chosen for on-device feasibility, fixed here before any run. Sub-cases cover base,
    missing-function, and missing-parameter turns (mirroring the BFCL multi-turn failure taxonomy).
  - *Per-turn agent loop:* each user turn runs an inner generate→parse→dispatch→feed-result loop,
    **≤ 5 tool-call rounds**, terminating when the model emits a final non-tool response.
  - *Oracle (scoring rule, fixed in advance):* after each turn's inner loop, the sandbox state is
    compared to that turn's **expected post-turn state** (state-diff; "unchanged" is the expected
    state for clarification / missing-function / missing-parameter turns). A turn **succeeds** iff the
    post-turn state matches expected.
  - *Agentic-reliability metrics (per cell, reported as rates):*
    - **task_success_rate** = fraction of tasks for which every turn's state matches AND the task did
      not hit the step budget;
    - **turn_failure_rate** = (Σ failed turns) / (Σ turns) over attempted tasks;
    - **nontermination_rate** = fraction of tasks in which any turn's inner loop reached the ≤5-round
      budget without a terminating response (loop / non-termination).
  - *Contract:* `score_detailed(instance, trace) → {task_success, turn_failures, nonterminated}`;
    `score()` returns `task_success` for pipeline compatibility.

- **E.5.4 Composite decision (LOCKED).** The multi-step agentic metrics are reported as a SEPARATE
  agentic-reliability result and are **NOT** folded into the 5-dimension LegitOnEdge composite (no 6th
  dimension). Redefining the tool-use dimension via multi-step task-success MAY be reported only as a
  robustness variant, never as the main composite. (The composite was already judged over-claimed at
  5 dims; an unvalidated 6th re-opens that.)

- **E.5.5 Hypotheses.**
  - **H3 (expanded).** Pearson r between composite capability and composite reliability is recomputed
    over 5 models × 2 tiers × {math, reasoning} = **20 cells** (was 12). Same test: two-sided on
    Fisher-z-transformed r, weak-correlation threshold |r| < 0.5; reported with bootstrap 95% CI
    (B=10,000). The n-boost is for power context; if the CI still straddles 0.5, H3 remains reported
    as exploratory (not guaranteed to resolve).
  - **H4 (new, directional).** The two function-calling-tuned models achieve higher single-call BFCL
    (`live_simple`) tool-use accuracy than the three general-purpose models. Tested as the difference
    in mean tool-use accuracy (tuned vs general) with a bootstrap 95% CI; with only 2 vs 3 model
    families it is under-powered for an α-level test and is reported as a pre-registered **directional
    expectation with effect + CI**, not a confirmatory p-value. Registered here before any expansion-phase
    tool-use measurement to foreclose post-hoc selection.

- **E.5.6 Bounds / τ (unchanged).** All composite-normalization bounds (latency 2×60 s, energy
  2×2000 J, throughput ref 0.05 tps/min; amendment E.4 F-A) and the R(τ) τ-anchors carry over
  UNCHANGED and were not re-selected using any expansion-phase data. The H3 composite sensitivity spans
  (bounds ±50%, bound-basis {absolute-SLA, per-tier-relative}, weights {uniform, each profile}) carry
  over and now span the 20-cell set.

- **E.5.7** Anything not fixed above remains exploratory/post-hoc per the "Exploratory analyses" section above.
