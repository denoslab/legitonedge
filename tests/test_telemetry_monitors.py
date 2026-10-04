"""Energy-monitor unit tests. The monitors spawn `ssh ... tegrastats|nvidia-smi`
subprocesses; tests mock subprocess.Popen so they don't touch the network.

Design note: readline() must block for poll_ms between calls so the reader
thread accumulates real dt. We use a helper that injects a sleep into each
fake readline call matching the monitor's poll_ms setting.
"""
from unittest.mock import patch, MagicMock
import time
import threading


def _make_fake_proc(lines: list[bytes], delay_s: float = 0.1):
    """Return a fake Popen-like object whose stdout.readline() yields each line
    after sleeping ``delay_s``, then returns b'' forever."""
    lock = threading.Lock()
    idx = [0]

    def readline():
        with lock:
            i = idx[0]
            idx[0] += 1
        if i < len(lines):
            time.sleep(delay_s)
            return lines[i]
        return b""

    proc = MagicMock()
    proc.stdout.readline = readline
    proc.poll.return_value = None
    return proc


def test_tegrastats_energy_monitor_integrates_joules():
    from legit_edge.telemetry import TegraStatsEnergyMonitor
    fake_lines = [b"RAM 1024/8000MB VDD_IN 5000mW\n"] * 5
    fake_proc = _make_fake_proc(fake_lines, delay_s=0.1)
    with patch("subprocess.Popen", return_value=fake_proc):
        m = TegraStatsEnergyMonitor(ssh_target="user@192.0.2.10", poll_ms=100)
        m.start()
        time.sleep(0.7)
        j = m.stop()
    # 5 samples × ~0.1 s × 5 W ≈ 2.5 J; loose tolerance for CI timing
    assert 1.0 < j < 5.0


def test_nvidiasmi_energy_monitor_integrates_joules():
    from legit_edge.telemetry import NvidiaSmiEnergyMonitor
    fake_lines = [b"45.0, 60\n"] * 10
    fake_proc = _make_fake_proc(fake_lines, delay_s=0.1)
    with patch("subprocess.Popen", return_value=fake_proc):
        m = NvidiaSmiEnergyMonitor(ssh_target="user@192.0.2.20", poll_ms=100)
        m.start()
        time.sleep(1.2)
        j = m.stop()
    # 10 samples × ~0.1 s × 45 W ≈ 45 J; loose window
    assert 20.0 < j < 70.0


def test_monitor_snap_returns_running_total():
    """snap() returns the cumulative J at call time; two successive snaps grow."""
    from legit_edge.telemetry import TegraStatsEnergyMonitor
    fake_lines = [b"VDD_IN 5000mW\n"] * 100
    fake_proc = _make_fake_proc(fake_lines, delay_s=0.05)
    with patch("subprocess.Popen", return_value=fake_proc):
        m = TegraStatsEnergyMonitor(ssh_target="x", poll_ms=50)
        m.start()
        time.sleep(0.3)
        j1 = m.snap()
        time.sleep(0.3)
        j2 = m.snap()
        m.stop()
    assert j2 > j1 > 0


def test_monitor_current_temp_c_populated_from_tegrastats():
    """tegrastats lines with CPU@/GPU@ populate self._last_temp_c via the parser."""
    from legit_edge.telemetry import TegraStatsEnergyMonitor
    fake_lines = [b"CPU@50C GPU@55C VDD_IN 3000mW\n"] * 3
    fake_proc = _make_fake_proc(fake_lines, delay_s=0.02)
    with patch("subprocess.Popen", return_value=fake_proc):
        m = TegraStatsEnergyMonitor(ssh_target="x", poll_ms=20)
        m.start()
        time.sleep(0.2)
        t = m.current_temp_c()
        m.stop()
    assert t == 55.0  # max of CPU/GPU


def test_monitor_current_temp_c_populated_from_nvidiasmi():
    from legit_edge.telemetry import NvidiaSmiEnergyMonitor
    fake_lines = [b"45.0, 72\n"] * 3
    fake_proc = _make_fake_proc(fake_lines, delay_s=0.02)
    with patch("subprocess.Popen", return_value=fake_proc):
        m = NvidiaSmiEnergyMonitor(ssh_target="x", poll_ms=20)
        m.start()
        time.sleep(0.2)
        t = m.current_temp_c()
        m.stop()
    assert t == 72.0
