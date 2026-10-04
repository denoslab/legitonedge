"""TestRunner: run one cell (model x tier x workload) at a chosen mode."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import datetime as _dt
import time
import json
from .adapter import ModelAdapter
from .workload import Workload, MultiTurnToolWorkload, CONFIDENCE_SUFFIX, parse_confidence
from .metrics import LatencyMetric, ThroughputStabilityMetric, EnergyMetric, MetricResult
from .telemetry import MockEnergyMonitor
from .tool_sandbox import ToolSandbox, parse_tool_calls, build_tool_system_prompt


@dataclass
class CellResult:
    cell_id: str
    mode: Literal["megaquick", "standard"]
    thermal: Literal["MAXN", "throttled"]
    capability: float
    latency: MetricResult
    throughput: MetricResult
    energy: MetricResult
    per_instance: list[dict] = field(default_factory=list)
    # Cell-level GPU-domain (overhead-subtracted) dynamic energy, integrated
    # from the compute rail (Jetson VDD_CPU_GPU_CV). None when the monitor exposes no
    # secondary channel (e.g. Spark nvidia-smi is already GPU-domain; older captures).
    energy_gpu_domain_j: float | None = None


def run_cell(
    *,
    adapter: ModelAdapter,
    workload: Workload,
    n_instances: int,
    mode: Literal["megaquick", "standard"],
    thermal: Literal["MAXN", "throttled"],
    confidence: bool = False,
    gen_tokens: int = 512,
    energy_monitor=None,
) -> CellResult:
    """Run one (model x tier x workload) cell. Returns per-instance traces + metric summaries."""
    energy_monitor = energy_monitor or MockEnergyMonitor()
    instances = workload.instances()[:n_instances]
    md = adapter.metadata()
    cell_id = f"{md.name}|{md.tier}|{workload.name}|{mode}|{thermal}"
    per_inst: list[dict] = []
    energy_monitor.start()
    t_start = time.perf_counter()
    for idx, inst in enumerate(instances):
        joules_before = energy_monitor.snap() if hasattr(energy_monitor, "snap") else 0.0
        t_before = time.perf_counter()
        temp_c = (
            energy_monitor.current_temp_c()
            if hasattr(energy_monitor, "current_temp_c") else None
        )
        timestamp = _dt.datetime.now(_dt.timezone.utc).isoformat()
        user_prompt = inst.input + (CONFIDENCE_SUFFIX if confidence else "")
        resp = adapter.generate(system="", user=user_prompt, expected=inst.target, thermal=thermal, max_tokens=gen_tokens)
        joules_after = energy_monitor.snap() if hasattr(energy_monitor, "snap") else 0.0
        elapsed_interval = max(1e-9, time.perf_counter() - t_before)
        # Trust adapter-reported latency (Ollama uses wall-clock; Mock uses persona-sampled
        # lognormal). The wall-clock-around-the-call would be ~zero for Mock.
        dt = resp.latency_s
        score = workload.score(inst, resp.text)
        # workloads may stash grader-side audit detail under instance.raw or via a side channel;
        # left as {} for now. BFCL grader output detail could go here.
        score_meta: dict = {}
        joules_interval = max(0.0, joules_after - joules_before)
        power_W_mean = joules_interval / elapsed_interval
        # Ollama exposes no token logprobs on the aarch64 builds we target, so
        # AdapterResponse has no logprobs field; record None for schema stability.
        logprobs_val = None
        confidence_val = parse_confidence(resp.text) if confidence else None
        per_inst.append({
            "instance_index": idx,
            "input": inst.input,
            "target": inst.target,
            "output": resp.text,
            "score": score,
            "score_metadata": score_meta,
            "latency_s": dt,
            "output_tokens": resp.output_tokens,
            "input_tokens": resp.input_tokens,
            "logprobs": logprobs_val,
            "confidence": confidence_val,
            "joules_interval": joules_interval,
            "power_W_mean_interval": power_W_mean,
            "timestamp": timestamp,
            "temperature_c": temp_c,
        })
    total_s = time.perf_counter() - t_start
    # Snap the GPU-domain rail before stop() (cell-level; not per-instance).
    total_j_gpu = energy_monitor.snap_secondary() if hasattr(energy_monitor, "snap_secondary") else None
    total_j = energy_monitor.stop()

    capability = sum(p["score"] for p in per_inst) / max(1, len(per_inst))

    latencies = [p["latency_s"] for p in per_inst]
    lat = LatencyMetric().compute(latencies)

    total_tokens = sum(p["output_tokens"] for p in per_inst)
    tps_avg = total_tokens / total_s if total_s else 0.0
    series: list[tuple[float, float]] = []
    cum_t = 0.0
    cum_tok = 0
    for p in per_inst:
        cum_t += p["latency_s"]
        cum_tok += p["output_tokens"]
        if cum_t > 0:
            series.append((cum_t, cum_tok / cum_t))
    if len(series) >= 2:
        tput = ThroughputStabilityMetric().compute(series)
    else:
        tput = MetricResult(
            name="throughput_stability",
            value={"slope_tps_per_min": 0.0, "mean_tps": tps_avg},
            n=len(series),
        )

    energy_samples = [(total_j / max(1, len(per_inst)), p["score"]) for p in per_inst]
    energy = EnergyMetric().compute(energy_samples)

    return CellResult(
        cell_id=cell_id,
        mode=mode,
        thermal=thermal,
        capability=capability,
        latency=lat,
        throughput=tput,
        energy=energy,
        per_instance=per_inst,
        energy_gpu_domain_j=total_j_gpu,
    )


def run_multiturn_cell(
    *,
    adapter: ModelAdapter,
    workload: MultiTurnToolWorkload,
    n_instances: int,
    mode: Literal["megaquick", "standard"],
    thermal: Literal["MAXN", "throttled"],
    gen_tokens: int = 512,
    step_budget: int = 5,
    energy_monitor=None,
) -> CellResult:
    """Drive each multi-turn task through a live ReAct loop, then score by replay.

    For each task: seed a fresh ``ToolSandbox`` from ``initial_state`` and serialize a
    growing text conversation. Each user turn runs an inner loop of up to ``step_budget``
    rounds of ``adapter.generate`` (system = ``build_tool_system_prompt()``, user = the
    conversation-so-far text). The assistant reply is parsed with ``parse_tool_calls``:
    an empty parse means the model gave its final answer (``terminated=True``, break);
    otherwise every parsed call is dispatched by name through the sandbox, the JSON
    results are appended back into the conversation, and the loop continues. Exhausting
    ``step_budget`` while still emitting calls yields ``terminated=False``.

    The produced ``trace = {"turns": [{"calls", "terminated"}, ...]}`` is scored by
    ``workload.score_detailed`` (the 2.2a state-diff oracle). Per-task latency and tokens
    are the SUM over every generate call in the task; energy is snapped once around the
    whole task (before turn 1, after the last turn). The returned ``CellResult`` mirrors
    ``run_cell``'s schema exactly so report/analysis stays uniform; ``score`` is the
    task's ``task_success`` and ``score_metadata`` is the full ``score_detailed`` dict.
    """
    energy_monitor = energy_monitor or MockEnergyMonitor()
    instances = workload.instances()[:n_instances]
    md = adapter.metadata()
    cell_id = f"{md.name}|{md.tier}|{workload.name}|{mode}|{thermal}"
    system_prompt = build_tool_system_prompt()
    per_inst: list[dict] = []
    energy_monitor.start()
    t_start = time.perf_counter()
    for idx, inst in enumerate(instances):
        spec = json.loads(inst.input)
        initial_state = spec.get("initial_state", {})
        user_turns = spec.get("turns", [])
        sandbox = ToolSandbox(initial_state=initial_state)

        joules_before = energy_monitor.snap() if hasattr(energy_monitor, "snap") else 0.0
        t_before = time.perf_counter()
        temp_c = (
            energy_monitor.current_temp_c()
            if hasattr(energy_monitor, "current_temp_c") else None
        )
        timestamp = _dt.datetime.now(_dt.timezone.utc).isoformat()

        conversation: list[str] = []
        trace_turns: list[dict] = []
        task_latency = 0.0
        task_out_tokens = 0
        task_in_tokens = 0
        for user_msg in user_turns:
            conversation.append(f"User: {user_msg}")
            executed_calls: list[dict] = []
            terminated = False
            for _ in range(step_budget):
                convo_text = "\n".join(conversation) + "\nAssistant:"
                resp = adapter.generate(
                    system=system_prompt, user=convo_text, expected=None,
                    thermal=thermal, max_tokens=gen_tokens,
                )
                task_latency += resp.latency_s
                task_out_tokens += resp.output_tokens
                task_in_tokens += resp.input_tokens
                conversation.append(f"Assistant: {resp.text}")
                calls = parse_tool_calls(resp.text)
                if not calls:
                    # No tool call -> the model gave its final answer for this turn.
                    terminated = True
                    break
                results = [sandbox.dispatch(c) for c in calls]
                executed_calls.extend(calls)
                conversation.append(f"Tool results: {json.dumps(results)}")
            # If the loop fell through without ever taking the no-call branch, the model
            # was still emitting calls at the step budget -> non-termination.
            trace_turns.append({"calls": executed_calls, "terminated": terminated})

        joules_after = energy_monitor.snap() if hasattr(energy_monitor, "snap") else 0.0
        elapsed_interval = max(1e-9, time.perf_counter() - t_before)
        trace = {"turns": trace_turns}
        detail = workload.score_detailed(inst, trace)
        joules_interval = max(0.0, joules_after - joules_before)
        power_W_mean = joules_interval / elapsed_interval
        per_inst.append({
            "instance_index": idx,
            "input": inst.input,
            "target": inst.target,
            "output": json.dumps(trace),
            "score": detail["task_success"],
            "score_metadata": detail,
            "latency_s": task_latency,
            "output_tokens": task_out_tokens,
            "input_tokens": task_in_tokens,
            "logprobs": None,
            "confidence": None,
            "joules_interval": joules_interval,
            "power_W_mean_interval": power_W_mean,
            "timestamp": timestamp,
            "temperature_c": temp_c,
        })
    total_s = time.perf_counter() - t_start
    # Snap the GPU-domain rail before stop() (cell-level; not per-instance).
    total_j_gpu = energy_monitor.snap_secondary() if hasattr(energy_monitor, "snap_secondary") else None
    total_j = energy_monitor.stop()

    capability = sum(p["score"] for p in per_inst) / max(1, len(per_inst))

    latencies = [p["latency_s"] for p in per_inst]
    lat = LatencyMetric().compute(latencies)

    total_tokens = sum(p["output_tokens"] for p in per_inst)
    tps_avg = total_tokens / total_s if total_s else 0.0
    series: list[tuple[float, float]] = []
    cum_t = 0.0
    cum_tok = 0
    for p in per_inst:
        cum_t += p["latency_s"]
        cum_tok += p["output_tokens"]
        if cum_t > 0:
            series.append((cum_t, cum_tok / cum_t))
    if len(series) >= 2:
        tput = ThroughputStabilityMetric().compute(series)
    else:
        tput = MetricResult(
            name="throughput_stability",
            value={"slope_tps_per_min": 0.0, "mean_tps": tps_avg},
            n=len(series),
        )

    energy_samples = [(total_j / max(1, len(per_inst)), p["score"]) for p in per_inst]
    energy = EnergyMetric().compute(energy_samples)

    return CellResult(
        cell_id=cell_id,
        mode=mode,
        thermal=thermal,
        capability=capability,
        latency=lat,
        throughput=tput,
        energy=energy,
        per_instance=per_inst,
        energy_gpu_domain_j=total_j_gpu,
    )
