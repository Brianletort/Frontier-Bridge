"""Detect parsers tested against fixture outputs — the Linux/NVIDIA path has no
live hardware here, so these fixtures are the only pre-hardware safety net."""

from frontier_bridge.detect import linux_nvidia, macos
from frontier_bridge.validation import validate_instance

NVIDIA_SMI_FIXTURE = (
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 580.65.06, 12.0\n"
)

MEMINFO_FIXTURE = """MemTotal:       65536000 kB
MemFree:        12345678 kB
MemAvailable:   23456789 kB
"""

LSCPU_FIXTURE = """Architecture:                       x86_64
CPU(s):                             32
Model name:                         AMD Ryzen Threadripper PRO 7975WX
NUMA node(s):                       1
"""

LSBLK_FIXTURE = """nvme0n1 disk 3840755982336 nvme
sda disk 1000204886016 sata
"""

SP_HARDWARE_FIXTURE = """{
  "SPHardwareDataType": [
    {
      "chip_type": "Apple M4 Max",
      "machine_model": "Mac16,6",
      "number_processors": "proc 16:12:4"
    }
  ]
}"""

_NULL_SSD_MEASURED = {
    "seq_read_gbps": None,
    "rand_read_4k_iops": None,
    "qd_used": None,
    "bench_tool": None,
}


def test_parse_nvidia_smi():
    gpus = linux_nvidia.parse_nvidia_smi(NVIDIA_SMI_FIXTURE)
    assert len(gpus) == 1
    assert gpus[0]["name"].startswith("NVIDIA RTX PRO 6000")
    assert gpus[0]["vram_gb"] == 102.6  # 97887 MiB in GB (decimal)
    assert gpus[0]["compute_cap"] == "12.0"


def test_parse_nvidia_smi_absent_tool():
    assert linux_nvidia.parse_nvidia_smi(None) == []
    assert linux_nvidia.parse_nvidia_smi("") == []


def test_parse_meminfo():
    assert linux_nvidia.parse_meminfo(MEMINFO_FIXTURE) == 67.1
    assert linux_nvidia.parse_meminfo(None) is None
    assert linux_nvidia.parse_meminfo("garbage") is None


def test_parse_lscpu():
    info = linux_nvidia.parse_lscpu(LSCPU_FIXTURE)
    assert info["cores"] == 32
    assert "Threadripper" in info["model"]
    assert info["numa_nodes"] == 1


def test_parse_lsblk_keeps_only_nvme():
    disks = linux_nvidia.parse_lsblk(LSBLK_FIXTURE)
    assert len(disks) == 1
    assert disks[0]["name"] == "nvme0n1"
    assert disks[0]["capacity_gb"] == 3840.8


def test_linux_build_profile_validates_and_never_guesses():
    profile = linux_nvidia.build_profile(
        gpus=linux_nvidia.parse_nvidia_smi(NVIDIA_SMI_FIXTURE),
        ram_gb=linux_nvidia.parse_meminfo(MEMINFO_FIXTURE),
        cpu_info=linux_nvidia.parse_lscpu(LSCPU_FIXTURE),
        nvme_disks=linux_nvidia.parse_lsblk(LSBLK_FIXTURE),
        ssd_measured=_NULL_SSD_MEASURED,
        os_version=None,
        kernel="6.8.0",
    )
    assert validate_instance(profile) == []
    assert profile["provenance"]["method"] == "detect"

    # Unprobed values must be null/unknown, never guessed.
    links_by_via = {link["via"]: link for link in profile["links"] if link["from"] == "sysram0"}
    pcie = links_by_via["pcie"]
    assert pcie["measured"]["h2d_gbps"] is None
    gds = [link for link in profile["links"] if link["via"] == "gds"]
    assert gds and gds[0]["available"] == "unknown"


def test_linux_build_profile_without_gpu_still_validates():
    profile = linux_nvidia.build_profile(
        gpus=[],
        ram_gb=64.0,
        cpu_info={"cores": 8, "model": None, "numa_nodes": None},
        nvme_disks=[],
        ssd_measured=_NULL_SSD_MEASURED,
        os_version=None,
        kernel=None,
    )
    assert validate_instance(profile) == []
    assert profile["profile_id"].startswith("nogpu")


def test_macos_parse_hardware_overview():
    info = macos.parse_hardware_overview(SP_HARDWARE_FIXTURE)
    assert info["chip"] == "Apple M4 Max"
    assert info["model"] == "Mac16,6"


def test_macos_parse_hardware_overview_bad_input():
    assert macos.parse_hardware_overview(None)["chip"] is None
    assert macos.parse_hardware_overview("not json")["chip"] is None


def test_macos_live_detect_validates():
    """Live end-to-end on this machine (disk bench skipped for speed)."""
    import platform

    if platform.system() != "Darwin":
        import pytest

        pytest.skip("macOS only")
    profile = macos.detect(run_disk_bench=False)
    assert validate_instance(profile) == []
    unified = [n for n in profile["nodes"] if n.get("class") == "unified"]
    assert unified and unified[0]["capacity_gb"] > 0
