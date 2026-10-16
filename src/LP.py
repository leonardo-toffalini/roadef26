import json
from collections import deque
from pathlib import Path

import networkx as nx
from docplex.mp.model import Model

from utils import load_demands, load_net, split_ratios

# path under data/, without the -net/-tm/-scenario suffix.
# toy is "toy/toy"; another instance is e.g. "setA/setA-01".
INSTANCE = "setA/setA-04"
WRITE_SRPATHS = True
# stop when ||z||_2^2 is within this fraction of the best bound
MIP_GAP = 0.05
# seconds; returns the best feasible solution found when this hits
TIME_LIMIT = 600

# hardcoded toy example use, no cli arg yet

# precompute link utilization vectors for each u, v pairs:
#   -> a_e^(uv) = unit flow fraction on edge e on the forwarding graph from u to v with the ECMP rule

# construct topologic closure of the original network G = (V, E) -> G' = (V, A)

# => about O(n^2) * 2 O(Dijkstra) time

# IP problem:
# goal:
#   min || z ||_2^2
#
# subject to:
#   sum_(qv in A) x_(qv)^(dt) - sum_(uq in A) x_(uq)^(dt) = 1 if q = s_d, -1 if q = t_d, 0 otherwise   forall q in V, forall d, forall t
#   sum_d sum_(uv in A) D^(dt) a_e^(uv) * x_(uv)^(dt) <= z_e    forall e in E, forall t
#   sum_(uv in A) x_(uv)^(dt) <= maxSeg    forall d, forall t
#   sum_d sum_(uv in A) | x_(uv)^(dt) - x_(uv)^(d(t-1)) | <= kappa(t)   forall t > 1


def load_instance(name: str = INSTANCE):
    base = Path(__file__).resolve().parents[1] / "data" / name
    graph = load_net(Path(f"{base}-net.json"))
    demands = load_demands(Path(f"{base}-tm.json"))
    scenario = json.loads(Path(f"{base}-scenario.json").read_text())
    return graph, demands, scenario


def utilization_vectors(
    graph: nx.DiGraph,
) -> dict[tuple[int, int], dict[tuple[int, int], float]]:
    """a_e^(uv): ECMP unit-flow fraction on edge e, for each ordered pair u != v."""
    nodes = list(graph.nodes)
    return {
        (u, v): split_ratios(graph, u, v)
        for u in nodes
        for v in nodes
        if u != v
    }


def topological_closure(graph: nx.DiGraph) -> nx.DiGraph:
    """G' = (V, A): same vertices, arc uv iff G has a directed path from u to v."""
    return nx.transitive_closure(graph, reflexive=None)


def ip_model(
    graph: nx.DiGraph,
    closure: nx.DiGraph,
    demands: list[dict],
    vectors: dict[tuple[int, int], dict[tuple[int, int], float]],
    scenario: dict,
) -> tuple[Model, dict, dict]:
    model = Model()
    model.parameters.mip.tolerances.mipgap = MIP_GAP
    model.set_time_limit(TIME_LIMIT)
    
    # define the to be minimized z upper bound on the MLU z
    z = model.continuous_var_dict(
        graph.edges, lb=0, key_format=lambda edge: f"z_{edge[0]}_{edge[1]}"
    )
    model.minimize(model.sumsq(z.values()))

    slots = max(len(demand["v"]) for demand in demands)

    # define the decision variables x_(uv)^(dt)
    x = model.binary_var_dict(
        ((u, v, d, t) for u, v in closure.edges for d in range(len(demands)) for t in range(slots)),
        key_format=lambda key: f"x_{key[0]}_{key[1]}_d{key[2]}_t{key[3]}",
    )

    # add the constraints:
    # sum_(qv in A) x_(qv)^(dt) - sum_(uq in A) x_(uq)^(dt) = 1 if q = s_d, -1 if q = t_d, 0 otherwise   forall q in V, forall d, forall t
    for d, demand in enumerate(demands):
        for t in range(slots):
            for q in graph.nodes:
                if q == demand["s"]:
                    balance = 1
                elif q == demand["t"]:
                    balance = -1
                else:
                    balance = 0
                out = model.sum(x[q, v, d, t] for v in closure.successors(q))
                inc = model.sum(x[u, q, d, t] for u in closure.predecessors(q))
                model.add_constraint(out - inc == balance)

    # sum_d sum_(uv in A) D^(dt) a_e^(uv) * x_(uv)^(dt) <= z_e    forall e in E, forall t
    for edge in graph.edges:
        touched = [
            (u, v, ratios[edge]) for (u, v), ratios in vectors.items() if edge in ratios
        ]
        for t in range(slots):
            load = model.sum(
                demands[d]["v"][t] * share * x[u, v, d, t]
                for u, v, share in touched
                for d in range(len(demands))
            )
            model.add_constraint(load <= z[edge])
    arcs = list(closure.edges)

    # sum_(uv in A) x_(uv)^(dt) <= maxSeg    forall d, forall t
    for d in range(len(demands)):
        for t in range(slots):
            model.add_constraint(model.sum(x[u, v, d, t] for u, v in arcs) <= scenario["max_segments"])

    # sum_d sum_(uv in A) | x_(uv)^(dt) - x_(uv)^(d(t-1)) | <= kappa(t)   forall t > 1
    budget = {step["t"]: step["value"] for step in scenario["budget"]}
    for t in range(1, slots):
        changed = model.sum(
            model.abs(x[u, v, d, t] - x[u, v, d, t - 1])
            for u, v in arcs
            for d in range(len(demands))
        )
        model.add_constraint(changed <= budget.get(t, 0))
    return model, z, x


def path_via(succ: dict[int, list[int]], source: int, target: int) -> list[int]:
    """inner nodes of a shortest path from source to target in succ."""
    previous = {source: None}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if node == target:
            break
        for nxt in succ.get(node, ()):
            if nxt not in previous:
                previous[nxt] = node
                queue.append(nxt)
    assert target in previous, (source, target)
    hops = []
    node = target
    while node != source:
        hops.append(node)
        node = previous[node]
    hops.reverse()
    return hops[:-1]


def waypoints(solution, x, closure: nx.DiGraph, source: int, target: int, d: int, t: int) -> list[int]:
    succ = {
        node: [
            v
            for v in closure.successors(node)
            if solution.get_value(x[node, v, d, t]) > 0.5
        ]
        for node in closure.nodes
    }
    return path_via(succ, source, target)


if __name__ == "__main__":
    # node 1 has two outgoing arcs; the cycle 1 -> 9 -> 1 is not the route to 5
    assert path_via({0: [1], 1: [9, 4], 9: [1], 4: [5]}, 0, 5) == [1, 4]
    graph, demands, scenario = load_instance()

    print("Precomputing the link utilization vectors...", flush=True)
    vectors = utilization_vectors(graph)
    for (u, v), ratios in vectors.items():
        if not ratios:
            continue
        leaving = sum(share for (src, _dst), share in ratios.items() if src == u)
        assert abs(leaving - 1.0) < 1e-9, (u, v, leaving)

    print("Building the topological closure...", flush=True)
    closure = topological_closure(graph)
    reached = {pair for pair, ratios in vectors.items() if ratios}
    assert set(closure.nodes) == set(graph.nodes)
    assert set(closure.edges) == reached

    print("Building the IP model...", flush=True)
    model, z, x = ip_model(graph, closure, demands, vectors, scenario)
    assert set(z) == set(graph.edges)

    print("Solving the IP model...", flush=True)
    solution = model.solve()
    if solution is None:
        status = getattr(model, "status", None) or getattr(model.solve_details, "status", "no solution")
        raise SystemExit(f"CPLEX finished without a feasible solution ({status})")

    slots = max(len(demand["v"]) for demand in demands)

    routes = []
    for d, demand in enumerate(demands):
        for t in range(slots):
            via = waypoints(solution, x, closure, demand["s"], demand["t"], d, t)
            routes.append({"d": d, "t": t, "w": via})
            print(routes[-1])

    if WRITE_SRPATHS:
        base = Path(__file__).resolve().parents[1] / "data" / INSTANCE
        path = Path(f"{base}-srpaths.json")
        path.write_text(json.dumps({"srpaths": routes}, indent=2) + "\n")
        print(path)

    for edge in graph.edges:
        peak = 0.0
        for t in range(slots):
            load = sum(
                demands[d]["v"][t] * ratios[edge] * solution.get_value(x[u, v, d, t])
                for (u, v), ratios in vectors.items()
                if edge in ratios
                for d in range(len(demands))
            )
            peak = max(peak, load)
        assert solution.get_value(z[edge]) + 1e-4 >= peak

    arcs = list(closure.edges)
    for d in range(len(demands)):
        for t in range(slots):
            used = sum(solution.get_value(x[u, v, d, t]) > 0.5 for u, v in arcs)
            assert used <= scenario["max_segments"], (d, t, used)

    budget = {step["t"]: step["value"] for step in scenario["budget"]}

    for t in range(1, slots):
        changed = sum(
            (solution.get_value(x[u, v, d, t]) > 0.5)
            != (solution.get_value(x[u, v, d, t - 1]) > 0.5)
            for u, v in arcs
            for d in range(len(demands))
        )
        assert changed <= budget.get(t, 0), (t, changed)

    print(
        INSTANCE,
        f"nodes={graph.number_of_nodes()}",
        f"edges={graph.number_of_edges()}",
        f"demands={len(demands)}",
        f"max_segments={scenario['max_segments']}",
        f"pairs={len(vectors)}",
        f"closure_arcs={closure.number_of_edges()}",
        f"objective={model.objective_value}",
        f"gap={model.solve_details.mip_relative_gap}",
    )
