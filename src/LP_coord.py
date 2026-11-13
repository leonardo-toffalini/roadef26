import json
import time
from pathlib import Path

import networkx as nx
from docplex.mp.model import Model

from LP import load_instance, topological_closure, utilization_vectors

# Same model as LP.py. The full IP is one binary per closure arc, demand, and
# time (about 980k on setA-04). Freeing one time slot still leaves |A|*|D|
# binaries, about half of that, because there are only two slots. Freeing one
# demand leaves |A|*T binaries, about 5k, which solves in a few seconds.
# Start from the direct routes and reoptimize one demand at a time.
INSTANCE = "setA/setA-04"
WRITE_SRPATHS = True
# per subproblem; the search stops early when the demand is solved
MIP_GAP = 0.0
# wall clock for the whole descent
TIME_LIMIT = 600
# wall clock for one demand; a feasible start is always installed
SUB_LIMIT = 10

Arc = tuple[int, int]


def hamming(left: list[Arc], right: list[Arc]) -> int:
    return len(set(left) ^ set(right))


def arcs_of(source: int, waypoints: list[int], target: int) -> list[Arc]:
    hops = [source, *waypoints, target]
    return list(zip(hops, hops[1:]))


def walk(chosen: list[Arc], source: int, target: int) -> list[int] | None:
    """Inner nodes of the unique source-target path in chosen, or None."""
    succ: dict[int, int] = {}
    for u, v in chosen:
        if u in succ:
            return None
        succ[u] = v
    hops = []
    node = source
    seen = 0
    while node != target:
        nxt = succ.get(node)
        if nxt is None:
            return None
        seen += 1
        if seen > len(chosen):
            return None
        node = nxt
        if node != target:
            hops.append(node)
    if seen != len(chosen):
        return None
    return hops


def descent(
    graph: nx.DiGraph,
    closure: nx.DiGraph,
    demands: list[dict],
    vectors: dict[Arc, dict[Arc, float]],
    scenario: dict,
    time_limit: float = TIME_LIMIT,
) -> tuple[list[list[list[Arc]]], float]:
    """Coordinate descent over demands, starting from the direct arc s -> t.

    Each pass reoptimizes one demand with the others fixed and keeps the new
    route as soon as it lowers sum_e z_e^2. A demand whose deletion bound is
    below EPS cannot improve: removing its load entirely is at least as good
    as any reroute. A demand that comes back with no improvement twice is not
    tried again.
    """
    slots = max(len(demand["v"]) for demand in demands)
    edges = list(graph.edges)
    arcs = list(closure.edges)
    nodes = list(graph.nodes)
    n_dem = len(demands)
    max_segments = scenario["max_segments"]
    budget = {step["t"]: step["value"] for step in scenario["budget"]}
    routes: list[list[list[Arc]]] = [
        [[(demand["s"], demand["t"])] for _ in range(slots)] for demand in demands
    ]
    load = {edge: [0.0] * slots for edge in edges}

    def add(demand: int, slot: int, used: list[Arc], sign: float) -> None:
        volume = demands[demand]["v"][slot] * sign
        for arc in used:
            for edge, share in vectors[arc].items():
                load[edge][slot] += volume * share

    for demand, route in enumerate(routes):
        for slot, used in enumerate(route):
            add(demand, slot, used, 1.0)

    def objective() -> float:
        return sum(max(values) ** 2 for values in load.values())

    def removal_gain(demand: int) -> float:
        acc: dict[tuple[Arc, int], float] = {}
        for slot, used in enumerate(routes[demand]):
            volume = demands[demand]["v"][slot]
            for arc in used:
                for edge, share in vectors[arc].items():
                    key = (edge, slot)
                    acc[key] = acc.get(key, 0.0) + volume * share
        drop = 0.0
        for edge in {edge for edge, _slot in acc}:
            old = max(load[edge])
            peak = max(load[edge][slot] - acc.get((edge, slot), 0.0) for slot in range(slots))
            new = max(0.0, peak)
            drop += old * old - new * new
        return drop

    def spent() -> list[int]:
        totals = [0] * slots
        for slot in range(1, slots):
            totals[slot] = sum(
                hamming(route[slot - 1], route[slot]) for route in routes
            )
        return totals

    def solve_demand(demand: int, room: list[int], limit: float) -> list[list[Arc]] | None:
        dem = demands[demand]
        source, target = dem["s"], dem["t"]
        residual = {edge: values[:] for edge, values in load.items()}
        for slot, used in enumerate(routes[demand]):
            volume = dem["v"][slot]
            for arc in used:
                for edge, share in vectors[arc].items():
                    residual[edge][slot] -= volume * share
        use = [(u, v) for u, v in arcs if v != source and u != target]
        succ: dict[int, list[int]] = {}
        pred: dict[int, list[int]] = {}
        for u, v in use:
            succ.setdefault(u, []).append(v)
            pred.setdefault(v, []).append(u)
        touched: dict[Arc, list[tuple[int, int, float]]] = {}
        for u, v in use:
            for edge, share in vectors[(u, v)].items():
                touched.setdefault(edge, []).append((u, v, share))

        model = Model(log_output=False)
        model.parameters.read.datacheck = 0
        model.parameters.mip.tolerances.mipgap = MIP_GAP
        model.set_time_limit(limit)
        z = {edge: model.continuous_var(lb=0) for edge in edges}
        x = {(u, v, slot): model.binary_var() for u, v in use for slot in range(slots)}
        for slot in range(slots):
            for node in nodes:
                if node == source:
                    balance = 1.0
                elif node == target:
                    balance = -1.0
                else:
                    balance = 0.0
                outs = succ.get(node, ())
                ins = pred.get(node, ())
                if not outs and not ins and balance == 0.0:
                    continue
                model.add_constraint(
                    model.sum(x[node, v, slot] for v in outs)
                    - model.sum(x[u, node, slot] for u in ins)
                    == balance
                )
            model.add_constraint(
                model.sum(x[u, v, slot] for u, v in use) <= max_segments
            )
        for slot in range(1, slots):
            changed = model.sum(
                model.abs(x[u, v, slot] - x[u, v, slot - 1]) for u, v in use
            )
            model.add_constraint(changed <= room[slot])
        for edge in edges:
            for slot in range(slots):
                expr = residual[edge][slot]
                terms = touched.get(edge)
                if terms:
                    volume = dem["v"][slot]
                    expr += model.sum(volume * share * x[u, v, slot] for u, v, share in terms)
                model.add_constraint(expr <= z[edge])
        model.minimize(model.sumsq(list(z.values())))
        start = model.new_solution()
        for slot, used in enumerate(routes[demand]):
            picked = set(used)
            for u, v in use:
                start.add_var_value(x[u, v, slot], 1 if (u, v) in picked else 0)
        model.add_mip_start(start)
        try:
            solution = model.solve(log_output=False)
            if solution is None:
                return None
            proposed = []
            for slot in range(slots):
                chosen = [(u, v) for u, v in use if solution.get_value(x[u, v, slot]) > 0.5]
                via = walk(chosen, source, target)
                if via is None:
                    return None
                path = arcs_of(source, via, target)
                if len(path) > max_segments:
                    return None
                proposed.append(path)
            for slot in range(1, slots):
                if hamming(proposed[slot - 1], proposed[slot]) > room[slot]:
                    return None
            return proposed
        finally:
            model.end()

    def commit(demand: int, proposed: list[list[Arc]]) -> float:
        for slot, used in enumerate(routes[demand]):
            add(demand, slot, used, -1.0)
        routes[demand] = proposed
        for slot, used in enumerate(proposed):
            add(demand, slot, used, 1.0)
        return objective()

    def room_for(demand: int) -> list[int]:
        totals = spent()
        room = [0] * slots
        for slot in range(1, slots):
            own = hamming(routes[demand][slot - 1], routes[demand][slot])
            room[slot] = budget.get(slot, 0) - (totals[slot] - own)
        return room

    started = time.perf_counter()
    current = objective()
    # ponytail: two misses and the demand is dropped for the rest of the run.
    # Retry when its deletion bound rises if that starts leaving gain behind.
    misses = [0] * n_dem
    print(f"direct objective {current:.4f}", flush=True)
    rnd = 0
    while time.perf_counter() - started < time_limit:
        rnd += 1
        order = sorted(
            (demand for demand in range(n_dem) if misses[demand] < 2),
            key=lambda demand: (misses[demand], -removal_gain(demand)),
        )
        improved = False
        solved = 0
        for demand in order:
            if removal_gain(demand) <= 1e-3:
                continue
            left = time_limit - (time.perf_counter() - started)
            if left <= 0:
                break
            limit = min(SUB_LIMIT, left)
            t0 = time.perf_counter()
            proposed = solve_demand(demand, room_for(demand), limit)
            elapsed = time.perf_counter() - t0
            solved += 1
            if proposed is None:
                continue
            for slot, used in enumerate(routes[demand]):
                add(demand, slot, used, -1.0)
            for slot, used in enumerate(proposed):
                add(demand, slot, used, 1.0)
            gain = current - objective()
            for slot, used in enumerate(proposed):
                add(demand, slot, used, -1.0)
            for slot, used in enumerate(routes[demand]):
                add(demand, slot, used, 1.0)
            if gain <= 1e-3:
                # a full time limit with no gain is not a proof
                if elapsed < limit - 0.05:
                    misses[demand] += 1
                print(
                    f"  d={demand} gain {gain:.1f} {elapsed:.1f}s",
                    flush=True,
                )
                continue
            current = commit(demand, proposed)
            misses[demand] = 0
            improved = True
            print(
                f"pass {rnd} keep d={demand} obj {current:.4f} "
                f"gain {gain:.1f} {elapsed:.1f}s "
                f"elapsed {time.perf_counter() - started:.1f}s",
                flush=True,
            )
        if not improved:
            print(f"pass {rnd} no improvement ({solved} solved)", flush=True)
            break
    return routes, objective()


def audit(
    graph: nx.DiGraph,
    demands: list[dict],
    vectors: dict[Arc, dict[Arc, float]],
    scenario: dict,
    routes: list[list[list[Arc]]],
) -> float:
    slots = max(len(demand["v"]) for demand in demands)
    load = {edge: [0.0] * slots for edge in graph.edges}
    budget = {step["t"]: step["value"] for step in scenario["budget"]}
    for demand, route in enumerate(routes):
        source, target = demands[demand]["s"], demands[demand]["t"]
        for slot, used in enumerate(route):
            assert walk(used, source, target) is not None, (demand, slot)
            assert len(used) <= scenario["max_segments"], (demand, slot, len(used))
            volume = demands[demand]["v"][slot]
            for arc in used:
                for edge, share in vectors[arc].items():
                    load[edge][slot] += volume * share
    for slot in range(1, slots):
        changed = sum(hamming(route[slot - 1], route[slot]) for route in routes)
        assert changed <= budget.get(slot, 0), (slot, changed, budget.get(slot, 0))
    return sum(max(values) ** 2 for values in load.values())


def srpaths(routes: list[list[list[Arc]]], demands: list[dict]) -> list[dict]:
    rows = []
    for demand, route in enumerate(routes):
        source, target = demands[demand]["s"], demands[demand]["t"]
        for slot, used in enumerate(route):
            via = walk(used, source, target)
            assert via is not None
            rows.append({"d": demand, "t": slot, "w": via})
    return rows


def _check_toy() -> None:
    graph, demands, scenario = load_instance("toy/toy")
    vectors = utilization_vectors(graph)
    closure = topological_closure(graph)
    direct = [
        [[(demand["s"], demand["t"])] for _ in range(len(demand["v"]))]
        for demand in demands
    ]
    direct_obj = audit(graph, demands, vectors, scenario, direct)
    routes, obj = descent(
        graph, closure, demands, vectors, scenario, time_limit=30
    )
    checked = audit(graph, demands, vectors, scenario, routes)
    assert abs(checked - obj) < 1e-4, (checked, obj)
    assert obj <= direct_obj + 1e-6, (obj, direct_obj)


if __name__ == "__main__":
    assert walk([(0, 1), (1, 4), (4, 5)], 0, 5) == [1, 4]
    assert walk([(0, 1), (1, 9), (9, 1), (1, 4)], 0, 4) is None
    assert hamming([(0, 1), (1, 2)], [(0, 1), (2, 3)]) == 2
    print("toy check", flush=True)
    _check_toy()

    graph, demands, scenario = load_instance()
    print("Precomputing the link utilization vectors...", flush=True)
    vectors = utilization_vectors(graph)
    print("Building the topological closure...", flush=True)
    closure = topological_closure(graph)
    reached = {pair for pair, ratios in vectors.items() if ratios}
    assert set(closure.edges) == reached

    print("Coordinate descent...", flush=True)
    routes, obj = descent(graph, closure, demands, vectors, scenario)
    checked = audit(graph, demands, vectors, scenario, routes)
    assert abs(checked - obj) < 1e-3, (checked, obj)

    rows = srpaths(routes, demands)
    for row in rows:
        if row["w"]:
            print(row)

    if WRITE_SRPATHS:
        base = Path(__file__).resolve().parents[1] / "data" / INSTANCE
        path = Path(f"{base}-srpaths.json")
        path.write_text(json.dumps({"srpaths": rows}, indent=2) + "\n")
        print(path)

    slots = max(len(demand["v"]) for demand in demands)
    print(
        INSTANCE,
        f"nodes={graph.number_of_nodes()}",
        f"edges={graph.number_of_edges()}",
        f"demands={len(demands)}",
        f"slots={slots}",
        f"objective={obj}",
    )
