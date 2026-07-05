"""Hardware detection: emit hwprofile/v1 documents for the current machine.

Principle: unknown values are null/unknown, never guessed. Every measured
number records how it was measured.
"""

from __future__ import annotations

import platform
from typing import Any

from frontier_bridge.detect import linux_nvidia, macos


def detect_hardware(run_disk_bench: bool = True) -> dict[str, Any]:
    """Detect the current machine and return an hwprofile/v1 dict."""
    system = platform.system()
    if system == "Darwin":
        return macos.detect(run_disk_bench=run_disk_bench)
    if system == "Linux":
        return linux_nvidia.detect(run_disk_bench=run_disk_bench)
    raise NotImplementedError(
        f"Detection is not supported on {system!r} yet. "
        "Use the manual template in hardware_profiles/templates/ instead."
    )
