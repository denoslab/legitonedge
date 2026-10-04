# LegitOnEdge

Benchmark code for **"The Device Decides: Benchmarking the Reliability of Agentic SLMs at the Edge"**, NeurIPS 2026 Workshop on Small Language Models for Agentic Systems.

LegitOnEdge measures how reliably small language models (SLMs) behave as agents when they are served on real edge hardware. Each benchmark cell is one model on one device running one workload. For every cell the framework records capability (task accuracy) together with systems reliability: latency tails, throughput, measured energy, run-to-run determinism and verbalized-confidence calibration. The same models and prompts run on two devices, so each cell can be compared with its twin on the other device. The two devices also run different Ollama builds, so a cross-device difference reflects the hardware together with its inference build.

The Python package is named `legit_edge` and installs a command-line tool called `legit-edge`.

## Hardware and models

Two devices, both serving models through [Ollama](https://ollama.com) over the LAN:

| Device | Software stack | Energy instrumentation |
|---|---|---|
| NVIDIA Jetson Orin Nano Super (8 GB) | JetPack 6.2 (L4T R36.4.4), Ollama 0.9.6 | `tegrastats` streamed over SSH (VDD_IN whole-board rail, plus the VDD_CPU_GPU_CV compute rail) |
| NVIDIA DGX Spark (GB10) | Ollama 0.21.0 | `nvidia-smi --query-gpu=power.draw` streamed over SSH (GPU domain) |

On the Jetson, the MAXN regime is `sudo nvpmodel -m 2` (MAXN_SUPER on this board) followed by `sudo jetson_clocks`. The throttled regime keeps the same power mode but leaves clocks under the default DVFS governor (no `jetson_clocks`). Note that `jetson_clocks` does not persist across reboots.

Five models, all Q4_K_M quantized, with identical Ollama tags on both devices (see `configs/models.yaml`):

- `llama3.1:8b-instruct-q4_K_M`
- `qwen2.5:7b-instruct-q4_K_M`
- `phi3.5:3.8b-mini-instruct-q4_K_M`
- `hermes3:8b-llama3.1-q4_K_M`
- `llama3-groq-tool-use:8b-q4_K_M`

Workloads (see `configs/workloads.yaml`; all subsets are drawn with seed 42):

- **math**: GSM8K-Hard (`reasoning-machines/gsm-hard`)
- **reasoning**: three BIG-Bench Hard subtasks (`lukaemon/bbh`)
- **tooluse**: BFCL v3 `live_simple`, graded against the BFCL ground truth
- **tooluse_mt**: a deterministic, locally generated multi-step tool-use task set executed in a sandbox and graded by state diff

Two test modes are supported: `megaquick` (small subsets, minutes per cell) and `standard` (about 200 prompts per single-turn workload).

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repository>
cd legitonedge
uv sync --extra dev
uv run legit-edge --help
```

On Windows, set `PYTHONUTF8=1` before running the CLI so that the console can print its status symbols.

## Configuring the devices

`configs/tiers.yaml` defines the two devices ("tiers"). Hosts and SSH users come from environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `LEGIT_EDGE_JETSON_HOST` | Jetson hostname or IP | `jetson.local` |
| `LEGIT_EDGE_SPARK_HOST` | DGX Spark hostname or IP | `spark.local` |
| `LEGIT_EDGE_JETSON_SSH_USER` | SSH login used to stream `tegrastats` | unset (ssh uses your local username or `~/.ssh/config`) |
| `LEGIT_EDGE_SPARK_SSH_USER` | SSH login used to stream `nvidia-smi` | unset (same fallback) |
| `LEGIT_EDGE_OLLAMA_TIMEOUT` | Per-request timeout in seconds | `300` |

```bash
export LEGIT_EDGE_JETSON_HOST=192.0.2.10
export LEGIT_EDGE_JETSON_SSH_USER=myuser
export LEGIT_EDGE_SPARK_HOST=192.0.2.20
export LEGIT_EDGE_SPARK_SSH_USER=myuser
uv run legit-edge doctor
```

On each device, expose Ollama on the LAN (for example `OLLAMA_HOST=0.0.0.0 ollama serve`, or a systemd override). Energy telemetry runs `ssh <user>@<host> tegrastats` or `ssh <user>@<host> nvidia-smi ...` non-interactively, so set up key-based SSH login from the benchmarking machine to each device first (for example with `ssh-copy-id`).

## Running the benchmark

```bash
uv run legit-edge doctor                  # readiness report for both devices
uv run legit-edge pin datasets            # download and freeze the workload subsets (seed 42)
uv run legit-edge pin models jetson       # pull the five models onto a device via /api/pull
uv run legit-edge pin models spark
uv run legit-edge smoke jetson            # one-prompt round trip per model
uv run legit-edge run jetson --mode megaquick --thermal MAXN
uv run legit-edge run jetson --mode standard --thermal throttled
uv run legit-edge run spark  --mode standard --thermal MAXN
uv run legit-edge report --out results/<run-dir> --format md
```

`pin datasets` must run before `run` on a fresh clone. It writes the frozen subsets to `data/workloads/`, which is not tracked in git.

Useful `run` options:

- `--workloads math,reasoning,tooluse,tooluse_mt` selects workloads (default: all four).
- `--models <key,...>` restricts the run to some of the model keys in `configs/models.yaml`.
- `--repeat k` repeats each cell k times for run-to-run determinism.
- `--confidence` appends a verbalized-confidence request to each prompt, used for calibration (ECE).
- `--out <dir>` overrides the results directory (default: `results/<target>-<mode>-<timestamp>/`).

Each cell writes a JSON-LD summary (`<model>__<target>__<workload>.json`) and a per-prompt trace file (`.traces.jsonl`) with latency, output tokens, energy and temperature.

Standard runs are long: a few hours on the DGX Spark and well over half a day on the Jetson. Launch them in the background and monitor the log.

There is also a hardware-free mock target for trying the pipeline end to end: `uv run legit-edge demo` (after `pin datasets`) runs simulated cells and prints a summary table. Mock results are synthetic and are not used in the paper.

## Analysis scripts

The scripts in `scripts/analysis/` are post-hoc analyses over run directories produced by `legit-edge run`. Most take run directories as arguments and print Markdown (use `--out` to write a file). Run `uv run python scripts/analysis/<script>.py --help` for exact arguments. The main ones are:

| Script | Computes |
|---|---|
| `h3_systems_composite.py` | Per-cell systems reliability composite (latency, throughput, energy) per deployment scenario, and its correlation with capability |
| `composite_score.py` | Per-cell composite scores and Reliability Card data |
| `reliability_function.py` | Empirical reliability function R(τ) = Pr(latency ≤ τ) per cell |
| `determinism.py` | Run-to-run determinism over `--repeat k` runs |
| `reshoot_calibration.py`, `reshoot_calibration_ci.py` | Verbalized-confidence ECE (the confidence line is stripped before grading) with bootstrap CIs |
| `agentic_reliability.py` | Multi-step tool-use task success and turn-failure rates |
| `failure_taxonomy.py` | Failure-mode classification of saved outputs |
| `energy_per_token.py`, `energy_per_token_railcompare.py`, `dynamic_energy.py` | Energy per output token at whole-board and at matched compute-rail boundaries, and idle-subtracted energy |
| `output_stability_paired.py`, `v2_thermal_energy_diag.py`, `reshoot_h1.py` | Paired MAXN versus throttled comparisons on the Jetson |
| `extrapolation_error_crosstier.py`, `h2_bootstrap_ci.py` | Agreement between mega-quick and standard estimates |
| `reshoot_h3.py`, `v3_h3_groq_sensitivity.py` | The contrast composite that adds calibration and output stability to the systems dimensions, and its leave-one-model-out sensitivity |

Two scripts (`h2_bootstrap_ci.py`, `v3_h3_groq_sensitivity.py`) work from derived tables in our own result archive rather than from raw run directories. They are included for transparency. Raw result data is not part of this repository.

`PRE_REGISTRATION.md` is the original analysis protocol, kept for the record.

## Tests

```bash
uv run pytest -q
```

The tests mock the network and SSH, so no hardware is required.

## Citation

```bibtex
@inproceedings{wu2026devicedecides,
  title     = {The Device Decides: Benchmarking the Reliability of Agentic {SLMs} at the Edge},
  author    = {Wu, Jiajun and Zhou, Jiayu and Drew, Steve},
  booktitle = {NeurIPS 2026 Workshop on Small Language Models for Agentic Systems},
  year      = {2026},
  note      = {To appear}
}
```

## License

MIT. See `LICENSE`.
