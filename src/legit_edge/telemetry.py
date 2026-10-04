"""Energy + thermal telemetry. INA260 preferred; tegrastats line parser for thermal."""
from __future__ import annotations
import collections
import re
import subprocess
import threading
import time
from dataclasses import dataclass


_CPU_TEMP = re.compile(r"CPU@(\d+(?:\.\d+)?)C", re.IGNORECASE)
_GPU_TEMP = re.compile(r"GPU@(\d+(?:\.\d+)?)C", re.IGNORECASE)
_VDD_IN = re.compile(r"VDD_IN (\d+)mW")
# GPU-domain (compute) + SoC rails. Each rail prints "<inst>mW/<avg>mW";
# we capture the instantaneous (first) value, matching _VDD_IN.
# NOTE: VDD_CPU_GPU_CV is the *combined* CPU+GPU+CV compute rail — the Orin Nano's
# INA3221 cannot isolate the GPU alone, so this over-counts dynamic GPU energy by the
# CPU share. It is still far closer to Spark's nvidia-smi GPU-domain power.draw than
# whole-board VDD_IN, hence its use as the overhead-subtracted dynamic-energy proxy.
_VDD_GPU = re.compile(r"VDD_CPU_GPU_CV (\d+)mW")
_VDD_SOC = re.compile(r"VDD_SOC (\d+)mW")


def parse_tegrastats_line(line: str) -> dict:
    out = {
        "cpu_temp_c": None, "gpu_temp_c": None,
        "power_mw": None, "power_gpu_mw": None, "power_soc_mw": None,
    }
    if m := _CPU_TEMP.search(line):
        out["cpu_temp_c"] = float(m.group(1))
    if m := _GPU_TEMP.search(line):
        out["gpu_temp_c"] = float(m.group(1))
    if m := _VDD_IN.search(line):
        out["power_mw"] = int(m.group(1))
    if m := _VDD_GPU.search(line):
        out["power_gpu_mw"] = int(m.group(1))
    if m := _VDD_SOC.search(line):
        out["power_soc_mw"] = int(m.group(1))
    return out


@dataclass
class MockEnergyMonitor:
    constant_watts: float = 10.0
    _t0: float = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def snap(self) -> float:
        """Joules accumulated so far, without stopping the monitor.

        Same formula as stop(): constant_watts × elapsed.
        """
        return float(self.constant_watts * (time.perf_counter() - self._t0))

    def snap_secondary(self) -> float:
        """GPU-domain (overhead-subtracted) joules proxy: a fixed fraction of primary.

        Real monitors derive this from the VDD_CPU_GPU_CV rail; Mock just returns
        0.6× the whole-board figure so the runner's hasattr('snap_secondary') path
        and downstream report field carry a plausible non-zero value in mock runs.
        """
        return float(0.6 * self.constant_watts * (time.perf_counter() - self._t0))

    def stop(self) -> float:
        return float(self.constant_watts * (time.perf_counter() - self._t0))


class INA260Monitor:
    """Real INA260 over I2C. Polls in a background thread; integrates W*dt to J."""

    def __init__(self, poll_hz: float = 50.0):
        self.poll_hz = poll_hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._joules = 0.0

    def start(self) -> None:
        self._joules = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self._joules

    def _loop(self) -> None:
        try:
            import board, busio
            from adafruit_ina260 import INA260
            i2c = busio.I2C(board.SCL, board.SDA)
            sensor = INA260(i2c)
        except Exception:
            return
        dt = 1.0 / self.poll_hz
        last = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            try:
                w = sensor.power / 1000.0
            except Exception:
                w = 0.0
            self._joules += w * (now - last)
            last = now
            time.sleep(dt)


class _SubprocessPowerMonitor:
    """Shared base: spawn an SSH+CLI subprocess streaming power lines, integrate W*dt -> J.

    Subclasses provide ``_cmd()`` and ``_parse_watts(line) -> float | None``.
    Temperature side-channel: subclasses may set ``self._last_temp_c`` during
    ``_parse_watts`` to allow the runner to snapshot per-instance temp without
    an extra parse pass.
    """

    def __init__(self, ssh_target: str, poll_ms: int = 100, max_buffer: int = 1000):
        self.ssh_target = ssh_target
        self.poll_ms = poll_ms
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._joules = 0.0
        # Secondary (GPU-domain) joules channel, integrated with the SAME
        # dt as the primary off the same source line. Stays 0.0 unless a subclass
        # overrides _parse_watts_secondary (default returns None).
        self._joules_secondary = 0.0
        self._last_t: float = 0.0
        self._last_temp_c: float | None = None
        self._samples: collections.deque = collections.deque(maxlen=max_buffer)

    def _cmd(self) -> list[str]:
        raise NotImplementedError

    def _parse_watts(self, line: str) -> float | None:
        raise NotImplementedError

    def _parse_watts_secondary(self, line: str) -> float | None:
        """GPU-domain watts off the same source line. Default: no secondary channel.

        Subclasses with a separable compute rail (TegraStatsEnergyMonitor) override
        this; Spark's nvidia-smi power.draw is already GPU-domain so it stays None.
        """
        return None

    def current_temp_c(self) -> float | None:
        """Latest temperature reading observed by the reader thread; None if unavailable."""
        return self._last_temp_c

    def start(self) -> None:
        self._joules = 0.0
        self._joules_secondary = 0.0
        self._last_t = time.perf_counter()
        self._stop.clear()
        self._proc = subprocess.Popen(
            self._cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._proc is not None
        first_sample = True
        while not self._stop.is_set():
            line_b = self._proc.stdout.readline()
            if not line_b:
                break
            decoded_line = line_b.decode("utf-8", errors="replace").strip()
            try:
                w = self._parse_watts(decoded_line)
            except Exception:
                w = None
            try:
                w2 = self._parse_watts_secondary(decoded_line)
            except Exception:
                w2 = None
            now = time.perf_counter()
            if w is not None:
                if first_sample:
                    # Reset the integration start point; ignore the spawn-to-first-sample delay.
                    self._last_t = now
                    first_sample = False
                else:
                    dt = now - self._last_t
                    self._last_t = now
                    if dt > 0:
                        self._joules += w * dt
                        self._samples.append((now, w))
                        # Same line, same timestamp -> integrate the GPU-domain rail
                        # with the identical dt (guarded; rail may be absent on a line).
                        if w2 is not None:
                            self._joules_secondary += w2 * dt

    def snap(self) -> float:
        """Read the current J running total without stopping the monitor."""
        return float(self._joules)

    def snap_secondary(self) -> float:
        """Read the GPU-domain J running total (0.0 if no secondary channel)."""
        return float(self._joules_secondary)

    def stop(self) -> float:
        self._stop.set()
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._thread:
            self._thread.join(timeout=2.0)
        return float(self._joules)


class TegraStatsEnergyMonitor(_SubprocessPowerMonitor):
    """Stream ``ssh <jetson> tegrastats --interval <ms>``, parse VDD_IN mW, integrate to J."""

    def _cmd(self) -> list[str]:
        return ["ssh", self.ssh_target, "tegrastats", "--interval", str(self.poll_ms)]

    def _parse_watts(self, line: str) -> float | None:
        out = parse_tegrastats_line(line)
        temps = [t for t in (out.get("cpu_temp_c"), out.get("gpu_temp_c")) if t is not None]
        if temps:
            self._last_temp_c = max(temps)
        mw = out.get("power_mw")
        return None if mw is None else float(mw) / 1000.0

    def _parse_watts_secondary(self, line: str) -> float | None:
        # GPU-domain proxy = VDD_CPU_GPU_CV (combined CPU+GPU+CV compute rail; the
        # Orin Nano's INA3221 cannot isolate the GPU alone — see _VDD_GPU note).
        mw = parse_tegrastats_line(line).get("power_gpu_mw")
        return None if mw is None else float(mw) / 1000.0


class NvidiaSmiEnergyMonitor(_SubprocessPowerMonitor):
    """Stream ``ssh <spark> nvidia-smi --query-gpu=power.draw,temperature.gpu --loop-ms=<ms>``, integrate to J."""

    def _cmd(self) -> list[str]:
        return [
            "ssh", self.ssh_target,
            "nvidia-smi", "--query-gpu=power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.poll_ms}",
        ]

    def _parse_watts(self, line: str) -> float | None:
        s = line.strip()
        if not s:
            return None
        parts = [p.strip() for p in s.split(",")]
        try:
            w = float(parts[0])
        except ValueError:
            return None
        if len(parts) >= 2:
            try:
                self._last_temp_c = float(parts[1])
            except ValueError:
                pass
        return w
