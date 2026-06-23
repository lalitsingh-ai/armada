"""Detect CPU architecture and Arm acceleration features.

The point of this module is evidence: the benchmark report should state *exactly* which
Arm matmul paths (i8mm, dotprod, sve2, bf16) were available on the machine that produced the
numbers, so the optimization claims are verifiable.
"""

from __future__ import annotations

import os
import platform
import re
from dataclasses import dataclass, field

# Arm64 /proc/cpuinfo "Features" flags that matter for ML inference, mapped to a readable label.
# Names follow the Linux kernel hwcap strings reported on aarch64.
ARM_FEATURE_LABELS: dict[str, str] = {
    "asimd": "NEON (Advanced SIMD)",
    "asimdhp": "FP16 arithmetic",
    "asimddp": "dotprod (INT8 dot product)",
    "i8mm": "i8mm (INT8 matrix multiply)",
    "bf16": "bf16 (BFloat16)",
    "sve": "SVE",
    "sve2": "SVE2",
    "svei8mm": "SVE INT8 matmul",
    "svebf16": "SVE BFloat16",
    "sme": "SME (Scalable Matrix Extension)",
    "sme2": "SME2",
}

# The flags whose presence we highlight as "fast LLM inference paths" on Arm.
ACCEL_FLAGS = ("i8mm", "asimddp", "bf16", "sve2", "sme2")


@dataclass
class CpuInfo:
    arch: str
    is_arm: bool
    model: str
    n_cores: int
    features: list[str] = field(default_factory=list)

    @property
    def accel_features(self) -> list[str]:
        """Subset of detected features that accelerate LLM matmul on Arm."""
        return [f for f in ACCEL_FLAGS if f in self.features]

    def summary(self) -> str:
        if not self.is_arm:
            return f"{self.arch} ({self.model}, {self.n_cores} cores) — x86 baseline"
        accel = ", ".join(self.accel_features) or "none detected"
        return f"{self.arch} ({self.model}, {self.n_cores} cores) — Arm accel: {accel}"

    def to_dict(self) -> dict:
        return {
            "arch": self.arch,
            "is_arm": self.is_arm,
            "model": self.model,
            "n_cores": self.n_cores,
            "features": self.features,
            "accel_features": self.accel_features,
        }


def _read_proc_cpuinfo() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def _parse_features(cpuinfo: str) -> list[str]:
    # aarch64 exposes a "Features" line; x86 uses "flags". We only care about the Arm set.
    match = re.search(r"^Features\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    if not match:
        return []
    raw = match.group(1).split()
    return [f for f in raw if f in ARM_FEATURE_LABELS]


def _parse_model(cpuinfo: str, arch: str) -> str:
    for key in ("model name", "Model name", "CPU part", "Hardware"):
        match = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        if match:
            return match.group(1).strip()
    return arch


def detect() -> CpuInfo:
    arch = platform.machine().lower()
    is_arm = arch in ("aarch64", "arm64")
    cpuinfo = _read_proc_cpuinfo()
    return CpuInfo(
        arch=arch,
        is_arm=is_arm,
        model=_parse_model(cpuinfo, arch),
        n_cores=os.cpu_count() or 1,
        features=_parse_features(cpuinfo),
    )
