from legit_edge.telemetry import MockEnergyMonitor, parse_tegrastats_line


def test_mock_energy_monitor_records_joules():
    m = MockEnergyMonitor(constant_watts=10.0)
    m.start()
    import time
    time.sleep(0.1)
    j = m.stop()
    assert 0.5 < j < 1.5


def test_parse_tegrastats_extracts_temp_and_power():
    line = "RAM 1024/8000MB CPU [10%@1500] EMC_FREQ 0% GR3D_FREQ 30% PLL@45C CPU@50C GPU@55C VDD_IN 5000mW"
    out = parse_tegrastats_line(line)
    assert out["cpu_temp_c"] == 50
    assert out["gpu_temp_c"] == 55
    assert out["power_mw"] == 5000


def test_parse_tegrastats_handles_lowercase_temp_fields():
    """Real JetPack 6.2 output uses lowercase cpu@/gpu@ — confirmed live on the Orin Nano."""
    line = ("05-28-2026 16:12:39 RAM 7328/7620MB CPU [21%@1651] GR3D_FREQ 0% "
            "cpu@44.812C soc2@43.687C soc0@44.093C gpu@45C tj@45C soc1@44.281C "
            "VDD_IN 4347mW/4347mW")
    from legit_edge.telemetry import parse_tegrastats_line
    out = parse_tegrastats_line(line)
    assert out["cpu_temp_c"] == 44.812
    assert out["gpu_temp_c"] == 45.0
    assert out["power_mw"] == 4347


def test_mock_energy_monitor_snap_returns_running_total():
    """Mock.snap() should return joules-so-far without stopping the monitor.

    Mirrors the _SubprocessPowerMonitor.snap() contract so the runner can call
    snap() on any monitor type uniformly.
    """
    import time
    from legit_edge.telemetry import MockEnergyMonitor
    m = MockEnergyMonitor(constant_watts=10.0)
    m.start()
    time.sleep(0.1)
    j1 = m.snap()
    time.sleep(0.1)
    j2 = m.snap()
    final = m.stop()
    assert j1 > 0
    assert j2 > j1
    assert final >= j2


def test_parse_tegrastats_extracts_gpu_and_soc_power_rails():
    """Parse the GPU-domain (VDD_CPU_GPU_CV) and SoC (VDD_SOC) rails.

    Live Orin Nano (JetPack R36.4.4) emits three rails on one line; we parse the
    instantaneous (first) mW of each, matching the existing VDD_IN convention.
    VDD_IN (whole board) must stay exactly as-is for backward compatibility.
    """
    line = ("05-28-2026 16:12:39 RAM 7328/7620MB cpu@44C gpu@45C "
            "VDD_IN 3876mW/3876mW VDD_CPU_GPU_CV 806mW/806mW VDD_SOC 1209mW/1209mW")
    out = parse_tegrastats_line(line)
    assert out["power_mw"] == 3876        # VDD_IN unchanged
    assert out["power_gpu_mw"] == 806     # VDD_CPU_GPU_CV instantaneous
    assert out["power_soc_mw"] == 1209    # VDD_SOC instantaneous


def test_parse_tegrastats_gpu_rail_absent_is_none():
    """A line without the compute rail (e.g. an older capture) yields None gracefully."""
    line = "RAM 1024/8000MB CPU [10%@1500] CPU@50C GPU@55C VDD_IN 5000mW"
    out = parse_tegrastats_line(line)
    assert out["power_mw"] == 5000
    assert out["power_gpu_mw"] is None
    assert out["power_soc_mw"] is None


def test_mock_energy_monitor_snap_secondary_is_positive_fraction():
    """MockEnergyMonitor.snap_secondary() returns a non-zero GPU-domain proxy.

    Lets the runner's hasattr('snap_secondary') path be exercised on Mock and
    keeps a plausible (< primary) dynamic-energy figure flowing through reports.
    """
    import time
    from legit_edge.telemetry import MockEnergyMonitor
    m = MockEnergyMonitor(constant_watts=10.0)
    m.start()
    time.sleep(0.1)
    sec = m.snap_secondary()
    prim = m.snap()
    assert sec > 0
    assert sec < prim  # GPU-domain is a fraction of whole-board


def test_subprocess_secondary_integration_math():
    """Hermetic: feed canned tegrastats lines through a stub _SubprocessPowerMonitor
    subclass and confirm the secondary (GPU-domain) joules channel integrates the
    VDD_CPU_GPU_CV rail with the same dt as the primary, independent of primary."""
    import time
    from legit_edge.telemetry import _SubprocessPowerMonitor, parse_tegrastats_line

    lines = [
        b"VDD_IN 4000mW/4000mW VDD_CPU_GPU_CV 1000mW/1000mW VDD_SOC 1200mW/1200mW\n",
        b"VDD_IN 4000mW/4000mW VDD_CPU_GPU_CV 1000mW/1000mW VDD_SOC 1200mW/1200mW\n",
        b"VDD_IN 4000mW/4000mW VDD_CPU_GPU_CV 1000mW/1000mW VDD_SOC 1200mW/1200mW\n",
    ]

    class _Stub(_SubprocessPowerMonitor):
        def __init__(self):
            super().__init__(ssh_target="x")
            self._idx = 0

        def _cmd(self):  # never spawned; we drive _loop's reader directly
            return ["true"]

        def _parse_watts(self, line: str):
            mw = parse_tegrastats_line(line).get("power_mw")
            return None if mw is None else float(mw) / 1000.0

        def _parse_watts_secondary(self, line: str):
            mw = parse_tegrastats_line(line).get("power_gpu_mw")
            return None if mw is None else float(mw) / 1000.0

    m = _Stub()

    class _FakeStdout:
        def __init__(self, rows):
            self._rows = list(rows)

        def readline(self):
            if not self._rows:
                return b""
            time.sleep(0.02)  # give dt a non-zero interval between samples
            return self._rows.pop(0)

    class _FakeProc:
        def __init__(self, rows):
            self.stdout = _FakeStdout(rows)

    m._proc = _FakeProc(lines)
    m._last_t = time.perf_counter()
    m._loop()
    j_prim = m.snap()
    j_sec = m.snap_secondary()
    assert j_prim > 0
    assert j_sec > 0
    # GPU rail is 1/4 of VDD_IN here -> secondary joules ~= primary/4 (same dt).
    assert abs(j_sec - j_prim / 4.0) < j_prim * 0.1


def test_subprocess_secondary_defaults_to_none_channel():
    """Base/Nvidia path: _parse_watts_secondary defaults to None, snap_secondary stays 0."""
    from legit_edge.telemetry import _SubprocessPowerMonitor
    m = _SubprocessPowerMonitor.__new__(_SubprocessPowerMonitor)
    # default method returns None regardless of input
    assert m._parse_watts_secondary("VDD_CPU_GPU_CV 999mW/999mW") is None
