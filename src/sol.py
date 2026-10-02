import itertools
import json
from pathlib import Path

import networkx as nx

from utils import down_edges, load_demands, load_net, loads


def routable(
    graph: nx.DiGraph,
    source: int,
    target: int,
    via: tuple[int, ...],
    reachable: dict[tuple[int, int], bool],
) -> bool:
    """True when each hop of ⟨source, *via, target⟩ has a path in graph."""
    hops = [source, *via, target]
    for src, dst in itertools.pairwise(hops):
        if (src, dst) not in reachable:
            reachable[(src, dst)] = nx.has_path(graph, src, dst)
        if not reachable[(src, dst)]:
            return False
    return True


def trial_value(
    graph: nx.DiGraph,
    demands: list[dict],
    scenario: Path,
    waypoints: dict[tuple[int, int], list[int]],
    t: int,
) -> float:
    """Max link load at time ``t`` under these waypoints."""
    return max(
        load
        for (*_edge, slot), load in loads(graph, demands, scenario, waypoints).items()
        if slot == t
    )


def solve_entry(
    graph: nx.DiGraph,
    demands: list[dict],
    scenario: Path,
    d: int,
    t: int,
    waypoints: dict[tuple[int, int], list[int]] | None = None,
) -> dict:
    """Best waypoints for demand ``d`` at time ``t``.

    Returns ``{"d", "t", "w"}``. Tries every ordered tuple of distinct
    intermediate nodes. A tuple's value is the max link load at time ``t``;
    other demands keep ``waypoints``. Shorter tuples are tried first, so a
    tie keeps the shorter one.
    """
    demand = demands[d]
    # one segment is ⟨s, t⟩; each waypoint adds a segment
    max_via = json.loads(scenario.read_text())["max_segments"] - 1
    candidates = [node for node in graph.nodes if node not in (demand["s"], demand["t"])]
    active = set(graph.edges) - down_edges(graph, scenario, t)
    live = graph.edge_subgraph(active).copy()
    reachable: dict[tuple[int, int], bool] = {}

    fixed = dict(waypoints or {})
    best_via: tuple[int, ...] | None = None
    best_value = float("inf")
    for length in range(max_via + 1):
        for via in itertools.permutations(candidates, length):
            if not routable(live, demand["s"], demand["t"], via, reachable):
                continue
            trial = dict(fixed)
            trial[(d, t)] = list(via)
            value = trial_value(graph, demands, scenario, trial, t)
            if value < best_value:
                best_value = value
                best_via = via
    if best_via is None:
        raise ValueError(f"demand {d} has no route at t={t}")
    return {"d": d, "t": t, "w": list(best_via)}


if __name__ == "__main__":
    # base = Path(__file__).resolve().parents[1] / "data" / "setA" / "setA-01"
    base = Path(__file__).resolve().parents[1] / "data" / "toy" / "toy"
    graph = load_net(Path(f"{base}-net.json"))
    demands = load_demands(Path(f"{base}-tm.json"))
    scenario = Path(f"{base}-scenario.json")
    # missing entries are direct ECMP; each solved entry is kept for the next one
    waypoints: dict[tuple[int, int], list[int]] = {}
    for d in range(2):
        for t in range(2):
            solved = solve_entry(graph, demands, scenario, d, t, waypoints)
            waypoints[(solved["d"], solved["t"])] = solved["w"]
            print(solved, trial_value(graph, demands, scenario, waypoints, solved["t"]))
