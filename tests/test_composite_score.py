import json, sys
from pathlib import Path
SCRIPTS = Path(__file__).parent.parent / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPTS.resolve()))


def _cell(d: Path, name: str, cap: float, p99: float, slope: float, jpc: float):
    (d / f"{name}.json").write_text(json.dumps({
        "cell_id": name, "mode": "standard", "thermal": "MAXN", "capability": cap,
        "metrics": {"latency": {"p50": p99/2, "p95": p99, "p99": p99},
                    "throughput_stability": {"slope_tps_per_min": slope},
                    "energy_per_correct": {"j_per_correct": jpc, "j_per_attempt": jpc*cap}},
    }))


def test_cell_composite_excludes_capability(tmp_path):
    from composite_score import cell_composite
    d = tmp_path
    _cell(d, "m__spark__math", cap=0.9, p99=6.0, slope=0.01, jpc=300.0)
    row = cell_composite(d / "m__spark__math.json", d_drift=0.0, calib_r=1.0,
                         weights={"latency":1,"throughput":1,"energy":1,"output_stability":1,"calibration":0})
    assert 0.0 <= row["reliability_composite"] <= 1.0
    assert row["capability"] == 0.9
    assert row["reliability_composite"] != row["capability"]


def test_pearson_with_ci_runs(tmp_path):
    from composite_score import pearson_with_ci
    import numpy as np
    x = np.array([0.1,0.2,0.3,0.4,0.5,0.6]); y = np.array([0.6,0.5,0.4,0.3,0.2,0.1])
    r, lo, hi = pearson_with_ci(x, y, b=200, seed=42)
    assert -1.0 <= lo <= r <= hi <= 1.0
    assert r < 0
