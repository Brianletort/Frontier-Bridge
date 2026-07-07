"""Fleet registry loading and fleet-wide planning (RFC 0005)."""

from frontier_bridge.fleet import fleet_plan, load_fleet
from frontier_bridge.planner.engine import generate_plan


def test_example_fleet_loads(repo_root):
    machines = load_fleet(repo_root, repo_root / "fleet" / "example.yaml")
    assert [m.name for m in machines] == ["mac-workstation", "spark", "linux-laptop"]
    assert machines[0].ssh is None
    assert machines[1].ssh == "you@your-spark"
    assert machines[2].hwprofile is None


def test_fleet_plan_matches_single_machine_planner(repo_root):
    """The fleet layer adds no new planning math: its verdict for a machine
    must equal the plain planner's verdict for that machine's profile."""
    machines = load_fleet(repo_root, repo_root / "fleet" / "example.yaml")
    rows = {
        r.machine: r
        for r in fleet_plan(repo_root, machines, "deepseek-v4-flash", "coding_agent", 32768)
    }
    single = generate_plan(
        repo_root=repo_root,
        model_id="deepseek-v4-flash",
        hardware_id="apple_m5_max_137gb_detected",
        workload="coding_agent",
        context_budget=32768,
    )
    assert rows["mac-workstation"].verdict == single["verdict"]
    assert rows["mac-workstation"].quant == single["inputs"]["modelprofile"].partition("/")[2]


def test_unprofiled_machines_are_listed_not_skipped(repo_root):
    machines = load_fleet(repo_root, repo_root / "fleet" / "example.yaml")
    rows = fleet_plan(repo_root, machines, "deepseek-v4-flash", "chat", 8192)
    unprofiled = [r for r in rows if r.machine == "linux-laptop"]
    assert unprofiled and unprofiled[0].verdict == "unprofiled"


def test_ranking_is_verdict_class_order(repo_root):
    machines = load_fleet(repo_root, repo_root / "fleet" / "example.yaml")
    rows = fleet_plan(repo_root, machines, "deepseek-v4-flash", "chat", 8192)
    order = {"recommended": 0, "experimental": 1, "not_recommended": 2}
    ranks = [order.get(r.verdict, 99) for r in rows]
    assert ranks == sorted(ranks)
