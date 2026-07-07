"""frontier doctor: readiness checks with injectable environment."""

from pathlib import Path

from frontier_bridge.doctor import FAIL, OK, WARN, run_checks, worst_status


def _which_all(binary: str) -> str:
    return f"/usr/bin/{binary}"


def _which_none(binary: str) -> None:
    return None


def _by_name(results):
    return {r.name: r for r in results}


def test_healthy_linux_box_is_all_ok():
    results = run_checks(
        models_dest=Path("models"),
        system="Linux",
        which=_which_all,
        disk_free_gb=500.0,
    )
    assert worst_status(results) == OK
    names = {r.name for r in results}
    assert {"python", "git", "fio", "nvidia-smi", "boltctl", "disk-space", "llama-server", "ssh"} <= names


def test_missing_tools_warn_with_fix_commands():
    results = _by_name(
        run_checks(
            models_dest=Path("models"),
            system="Linux",
            which=_which_none,
            disk_free_gb=500.0,
        )
    )
    assert results["fio"].status == WARN
    assert "apt install" in results["fio"].fix
    assert results["llama-server"].status == WARN
    assert "build_llama_cpp_cuda" in results["llama-server"].fix
    assert results["git"].status == FAIL


def test_llama_fix_is_platform_specific():
    linux = _by_name(
        run_checks(models_dest=Path("m"), system="Linux", which=_which_none, disk_free_gb=500.0)
    )
    mac = _by_name(
        run_checks(models_dest=Path("m"), system="Darwin", which=_which_none, disk_free_gb=500.0)
    )
    assert "build_llama_cpp_cuda" in linux["llama-server"].fix
    assert "brew install" in mac["llama-server"].fix


def test_nvidia_checks_only_on_linux():
    mac = {r.name for r in run_checks(Path("m"), system="Darwin", which=_which_all, disk_free_gb=500.0)}
    assert "nvidia-smi" not in mac
    assert "boltctl" not in mac


def test_disk_space_thresholds():
    tight = _by_name(run_checks(Path("m"), system="Linux", which=_which_all, disk_free_gb=100.0))
    assert tight["disk-space"].status == WARN
    empty = _by_name(run_checks(Path("m"), system="Linux", which=_which_all, disk_free_gb=20.0))
    assert empty["disk-space"].status == FAIL
    assert worst_status(list(empty.values())) == FAIL
