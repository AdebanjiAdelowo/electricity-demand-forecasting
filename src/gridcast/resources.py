"""Portable CPU-only resource admission; Linux memory sensors fail closed.

macOS admission remains the dispatch memory-pressure gate in
`experiments.fair_resource_pilot`. On Linux (including Colab) admission uses
`/proc/meminfo`, the process's cgroup memory limit/usage, optional pressure
stall information, disk space, CPU affinity/quota and accelerator detection.
Host RAM alone never admits a run: an unreadable container memory controller
blocks it. Thresholds are conservative engineering margins, not measurements.
"""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

MIB = 1024**2
SYSTEM_MARGIN_MIB = 1024
MIN_DISK_BYTES = 1024**3
PSI_SOME_AVG10_LIMIT = 10.0
MAX_THREADS = 2
UNLIMITED = 2**60
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GRIDCAST_JOBS",
)


class AdmissionBlocked(RuntimeError):
    pass


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def meminfo(root: Path = Path("/")) -> dict[str, int]:
    text = _read(root / "proc/meminfo") or ""
    out = {}
    for line in text.splitlines():
        name, _, value = line.partition(":")
        parts = value.split()
        if parts and parts[0].isdigit():
            out[name] = int(parts[0]) * (1024 if parts[1:] == ["kB"] else 1)
    return out


def process_rss(root: Path = Path("/"), pid: str = "self") -> int | None:
    for line in (_read(root / f"proc/{pid}/status") or "").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def _limit(text: str | None) -> int | None:
    value = (text or "").strip()
    if value == "max":
        return UNLIMITED
    return int(value) if value.isdigit() else None


def cgroup_memory(root: Path = Path("/")) -> dict:
    """Smallest limit along the process's cgroup ancestry; None if unreadable."""
    base = root / "sys/fs/cgroup"
    lines = (_read(root / "proc/self/cgroup") or "").splitlines()
    unified = next((ln.split("::", 1)[1] for ln in lines if ln.startswith("0::")), None)
    if unified is not None or (base / "memory.max").exists() or (base / "cgroup.controllers").exists():
        limits, usage = [], None
        node = Path(unified or "/")
        while True:
            directory = base / str(node).lstrip("/")
            found = _limit(_read(directory / "memory.max"))
            if found is not None:
                limits.append(found)
                usage = usage if usage is not None else _limit(_read(directory / "memory.current"))
            if node == node.parent:
                break
            node = node.parent
        if limits and usage is not None:
            return {"version": 2, "limit_bytes": min(limits), "usage_bytes": usage}
        if (base / "cgroup.controllers").exists() and unified in (None, "/"):
            # The root cgroup has no memory.max; a root-namespace view is unlimited.
            current = _limit(_read(base / "memory.current"))
            return {"version": 2, "limit_bytes": UNLIMITED, "usage_bytes": current, "root": True}
    v1 = base / "memory"
    limit, usage = _limit(_read(v1 / "memory.limit_in_bytes")), _limit(_read(v1 / "memory.usage_in_bytes"))
    if limit is not None and usage is not None:
        return {"version": 1, "limit_bytes": limit, "usage_bytes": usage}
    return {"version": None, "limit_bytes": None, "usage_bytes": None}


def psi_memory(root: Path = Path("/")) -> dict | None:
    out = {}
    for line in (_read(root / "proc/pressure/memory") or "").splitlines():
        kind, *fields = line.split()
        values = dict(f.split("=", 1) for f in fields if "=" in f)
        if "avg10" in values:
            out[f"{kind}_avg10"] = float(values["avg10"])
    return out or None


def cpu_info(root: Path = Path("/")) -> dict:
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    quota = None
    text = _read(root / "sys/fs/cgroup/cpu.max")
    if text and text.split()[0] != "max":
        allowed, period = text.split()[:2]
        quota = int(allowed) / int(period)
    model = next(
        (
            ln.split(":", 1)[1].strip()
            for ln in (_read(root / "proc/cpuinfo") or "").splitlines()
            if ln.startswith("model name")
        ),
        platform.processor() or None,
    )
    usable = [v for v in (os.cpu_count(), affinity, quota) if v]
    return {
        "logical": os.cpu_count(),
        "affinity": affinity,
        "cgroup_quota_cpus": quota,
        "usable_cpus": min(usable) if usable else None,
        "model": model,
    }


def accelerators(root: Path = Path("/"), environ=None) -> list[str]:
    environ = os.environ if environ is None else environ
    found = [f"/{p}" for p in ("dev/nvidia0", "dev/nvidiactl", "dev/accel0") if (root / p).exists()]
    if environ.get("COLAB_GPU", "") not in ("", "0"):
        found.append("COLAB_GPU")
    found += [k for k in ("COLAB_TPU_ADDR", "TPU_NAME") if environ.get(k)]
    return found


def runtime(environ=None) -> str:
    environ = os.environ if environ is None else environ
    if environ.get("COLAB_RELEASE_TAG") or environ.get("COLAB_BACKEND_VERSION"):
        return "colab"
    if environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "local"


def thread_environment(threads: int) -> dict[str, str]:
    """Child-process variables: bounded threads and no visible CUDA device."""
    if not isinstance(threads, int) or not 1 <= threads <= MAX_THREADS:
        raise ValueError(f"Threads must be between 1 and {MAX_THREADS}")
    return {**dict.fromkeys(THREAD_VARIABLES, str(threads)), "CUDA_VISIBLE_DEVICES": ""}


def thread_problems(environ, threads: int) -> list[str]:
    problems = []
    for name in THREAD_VARIABLES:
        value = environ.get(name)
        if value is None:
            problems.append(f"{name} unset")
        elif not value.isdigit() or int(value) > threads:
            problems.append(f"{name}={value} exceeds {threads}")
    return problems


def linux_snapshot(root: Path = Path("/"), environ=None, disk_path: Path | None = None) -> dict:
    environ = os.environ if environ is None else environ
    memory = meminfo(root)
    group = cgroup_memory(root)
    available = memory.get("MemAvailable")
    container = None
    if (
        group["limit_bytes"] is not None
        and group["limit_bytes"] < UNLIMITED
        and group["usage_bytes"] is not None
    ):
        container = max(0, group["limit_bytes"] - group["usage_bytes"])
    candidates = [v for v in (available, container) if v is not None]
    return {
        "sensor": "linux",
        "runtime": runtime(environ),
        "mem_total_bytes": memory.get("MemTotal"),
        "mem_available_bytes": available,
        "swap_total_bytes": memory.get("SwapTotal"),
        "cgroup": group,
        "container_available_bytes": container,
        "effective_available_bytes": min(candidates) if available is not None else None,
        "process_rss_bytes": process_rss(root),
        "psi_memory": psi_memory(root),
        "cpu": cpu_info(root),
        "accelerators": accelerators(root, environ),
        "disk_free_bytes": shutil.disk_usage(disk_path or Path.cwd()).free,
        "thread_environment": {k: environ.get(k) for k in (*THREAD_VARIABLES, "CUDA_VISIBLE_DEVICES")},
    }


def assess(record: dict, ceiling_mib: int, threads: int) -> dict:
    """READY / CONDITIONAL / BLOCKED; never relaxes the ceiling for a large host."""
    blocking, warnings = [], []
    if record.get("sensor") != "linux":
        blocking.append("no Linux resource record")
        return {"verdict": "BLOCKED", "blocking": blocking, "warnings": warnings}
    rss = record.get("process_rss_bytes")
    if rss is None:
        blocking.append("process RSS sensor unavailable")
        rss = 0
    required = max(0, ceiling_mib * MIB - rss) + SYSTEM_MARGIN_MIB * MIB
    if record.get("mem_available_bytes") is None:
        blocking.append("/proc/meminfo MemAvailable unavailable")
    if record["cgroup"].get("version") is None:
        blocking.append("container (cgroup) memory limit unverifiable")
    elif record["cgroup"].get("limit_bytes", 0) >= UNLIMITED:
        warnings.append("no cgroup memory limit found; host MemAvailable used as the binding sensor")
    available = record.get("effective_available_bytes")
    if available is not None and available < required:
        blocking.append(
            f"effective available memory {available / MIB:.0f} MiB < required {required / MIB:.0f} MiB "
            f"(ceiling {ceiling_mib} MiB minus current RSS plus {SYSTEM_MARGIN_MIB} MiB system margin)"
        )
    psi = record.get("psi_memory")
    if psi is None:
        warnings.append("Linux memory pressure-stall information unavailable")
    elif psi.get("some_avg10", 0.0) >= PSI_SOME_AVG10_LIMIT:
        blocking.append(f"memory pressure stall some avg10={psi['some_avg10']} >= {PSI_SOME_AVG10_LIMIT}")
    if record.get("disk_free_bytes", 0) < MIN_DISK_BYTES:
        blocking.append("less than 1 GiB free disk for results")
    if record.get("accelerators"):
        blocking.append(f"accelerator attached ({', '.join(record['accelerators'])}); select a CPU runtime")
    usable = record.get("cpu", {}).get("usable_cpus")
    if usable is None or usable < 1:
        blocking.append("no usable CPU count")
    elif usable < threads:
        warnings.append(f"{usable} usable CPU(s) for {threads} threads; timings may be oversubscribed")
    blocking += thread_problems(record.get("thread_environment") or {}, threads)
    verdict = "BLOCKED" if blocking else "CONDITIONAL" if warnings else "READY"
    return {
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "required_available_bytes": required,
    }


def linux_admit(record: dict, ceiling_mib: int, threads: int) -> dict:
    decision = assess(record, ceiling_mib, threads)
    record["admission"] = decision
    if decision["verdict"] == "BLOCKED":
        raise AdmissionBlocked("; ".join(decision["blocking"]))
    return decision
