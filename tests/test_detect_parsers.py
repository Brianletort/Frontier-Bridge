"""Detect parsers tested against fixture outputs — the Linux/NVIDIA path has no
live hardware here, so these fixtures are the only pre-hardware safety net."""

from frontier_bridge.detect import common, linux_nvidia, macos
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


def test_parse_os_release():
    text = 'NAME="Debian GNU/Linux"\nPRETTY_NAME="Debian GNU/Linux 13 (trixie)"\n'
    assert linux_nvidia.parse_os_release(text) == "Debian GNU/Linux 13 (trixie)"
    assert linux_nvidia.parse_os_release(None) is None
    assert linux_nvidia.parse_os_release("NAME=x\n") is None


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


GB10_NVIDIA_SMI_FIXTURE = "NVIDIA GB10, 122880, 580.95.05, 12.1\n"

PROC_VERSION_WSL2 = (
    "Linux version 6.6.87.2-microsoft-standard-WSL2 "
    "(root@builder) (gcc ...) #1 SMP ...\n"
)
PROC_VERSION_NATIVE = "Linux version 6.8.0-45-generic (buildd@lcy02) ...\n"

FIO_JSON_FIXTURE = """{
  "fio version": "fio-3.36",
  "jobs": [
    {
      "job options": {"iodepth": "32"},
      "read": {"bw_bytes": 7100000000, "iops": 6771.5}
    }
  ]
}"""

NVBANDWIDTH_FIXTURE = """nvbandwidth Version: v0.5
Running host_to_device_memcpy_ce.
SUM host_to_device_memcpy_ce 55.23
Running device_to_host_memcpy_ce.
SUM device_to_host_memcpy_ce 52.10
"""


def test_is_wsl2_from_proc_version():
    assert common.is_wsl2(PROC_VERSION_WSL2) is True
    assert common.is_wsl2(PROC_VERSION_NATIVE) is False


def test_parse_fio_json():
    measured = common.parse_fio_json(FIO_JSON_FIXTURE)
    assert measured["seq_read_gbps"] == 7.1
    assert measured["qd_used"] == 32
    assert "fio-3.36" in measured["bench_tool"]
    assert common.parse_fio_json(None) is None
    assert common.parse_fio_json("not json") is None
    assert common.parse_fio_json('{"jobs": []}') is None


def test_parse_nvbandwidth():
    measured = common.parse_nvbandwidth(NVBANDWIDTH_FIXTURE)
    assert measured["h2d_gbps"] == 55.2
    assert measured["d2h_gbps"] == 52.1
    assert measured["pinned"] is True
    assert common.parse_nvbandwidth("no sum lines") is None
    assert common.parse_nvbandwidth(None) is None


def test_unified_memory_heuristic():
    gb10 = linux_nvidia.parse_nvidia_smi(GB10_NVIDIA_SMI_FIXTURE)
    rtx = linux_nvidia.parse_nvidia_smi(NVIDIA_SMI_FIXTURE)
    assert linux_nvidia.is_unified_memory_system(gb10, "aarch64") is True
    assert linux_nvidia.is_unified_memory_system(rtx, "x86_64") is False
    # aarch64 alone is not enough — discrete stays the safe default.
    assert linux_nvidia.is_unified_memory_system(rtx, "aarch64") is False


def test_gb10_unified_profile_topology():
    profile = linux_nvidia.build_profile(
        gpus=linux_nvidia.parse_nvidia_smi(GB10_NVIDIA_SMI_FIXTURE),
        ram_gb=119.9,
        cpu_info={"cores": 20, "model": "GB10 Grace CPU", "numa_nodes": 1},
        nvme_disks=[{"name": "nvme0n1", "capacity_gb": 3840.8}],
        ssd_measured=_NULL_SSD_MEASURED,
        os_version=None,
        kernel="6.11.0",
        unified=True,
        machine="aarch64",
    )
    assert validate_instance(profile) == []
    memory_nodes = [n for n in profile["nodes"] if n["kind"] == "memory"]
    assert len(memory_nodes) == 1
    assert memory_nodes[0]["class"] == "unified"
    unified_links = [link for link in profile["links"] if link["via"] == "unified"]
    assert unified_links and unified_links[0]["available"] is True
    # No separate vram node and no gds link on coherent-memory systems.
    assert not any(n["id"].startswith("vram") for n in profile["nodes"])
    assert not any(link["via"] == "gds" for link in profile["links"])


def test_wsl2_profile_annotation_and_gds_false():
    profile = linux_nvidia.build_profile(
        gpus=linux_nvidia.parse_nvidia_smi(NVIDIA_SMI_FIXTURE),
        ram_gb=48.0,  # WSL-assigned, not host RAM
        cpu_info={"cores": 32, "model": None, "numa_nodes": 1},
        nvme_disks=[{"name": "sdc", "capacity_gb": 2000.0}],
        ssd_measured=_NULL_SSD_MEASURED,
        os_version=None,
        kernel="6.6.87.2-microsoft-standard-WSL2",
        wsl2=True,
        machine="x86_64",
    )
    assert validate_instance(profile) == []
    assert profile["provenance"]["virtualization"] == "wsl2"
    assert profile["profile_id"].endswith("wsl2_detected")
    gds = [link for link in profile["links"] if link["via"] == "gds"]
    assert gds and gds[0]["available"] is False


def test_pcie_probe_results_land_on_first_gpu_link():
    profile = linux_nvidia.build_profile(
        gpus=linux_nvidia.parse_nvidia_smi(NVIDIA_SMI_FIXTURE),
        ram_gb=64.0,
        cpu_info={"cores": 32, "model": None, "numa_nodes": 1},
        nvme_disks=[],
        ssd_measured=_NULL_SSD_MEASURED,
        os_version=None,
        kernel="6.8.0",
        pcie_measured={"h2d_gbps": 55.2, "d2h_gbps": 52.1, "pinned": True},
        machine="x86_64",
    )
    assert validate_instance(profile) == []
    pcie = [
        link for link in profile["links"]
        if link["via"] == "pcie" and link["to"] == "vram0"
    ]
    assert pcie[0]["measured"]["h2d_gbps"] == 55.2
    assert pcie[0]["measured"]["pinned"] is True


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
