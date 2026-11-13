import json
import time
from collections import deque
from pathlib import Path

import cplex
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
TIME_LIMIT = 60 * 30
# False keeps the docplex model in ip_model, built one expression at a time.
# True is the same constraints, passed to CPLEX as batches of coefficients.
BATCH_BUILD = True

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


class _XCols:
    def __init__(self, arc_index: dict[tuple[int, int], int], n_dem: int, slots: int):
        self.arc_index = arc_index
        self.n_dem = n_dem
        self.slots = slots

    def __getitem__(self, key: tuple[int, int, int, int]) -> int:
        u, v, d, t = key
        return (self.arc_index[u, v] * self.n_dem + d) * self.slots + t


class _BatchSolution:
    def __init__(self, values: list[float]):
        self._values = values

    def get_value(self, index: int) -> float:
        return self._values[index]


class _BatchDetails:
    def __init__(self, gap: float):
        self.mip_relative_gap = gap


class _BatchModel:
    def __init__(self, problem: cplex.Cplex):
        self._problem = problem
        self.objective_value = None
        self.status = "not solved"
        self.solve_details = _BatchDetails(0.0)

    def solve(self) -> _BatchSolution | None:
        self._problem.solve()
        self.status = self._problem.solution.get_status_string()
        if not self._problem.solution.is_primal_feasible():
            return None
        self.objective_value = self._problem.solution.get_objective_value()
        try:
            gap = self._problem.solution.MIP.get_mip_relative_gap()
        except cplex.exceptions.CplexSolverError:
            gap = 0.0
        self.solve_details = _BatchDetails(gap)
        return _BatchSolution(self._problem.solution.get_values())


def _add_vars(problem: cplex.Cplex, count: int, lb: float, ub: float, kind: str) -> None:
    chunk = 500_000
    for start in range(0, count, chunk):
        size = min(chunk, count - start)
        problem.variables.add(lb=[lb] * size, ub=[ub] * size, types=kind * size)


def _add_rows(
    problem: cplex.Cplex,
    inds: list[list[int]],
    vals: list[list[float]],
    senses: list[str],
    rhs: list[float],
) -> None:
    if not rhs:
        return
    problem.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind, val) for ind, val in zip(inds, vals)],
        senses=senses,
        rhs=rhs,
    )


def ip_model_batch(
    graph: nx.DiGraph,
    closure: nx.DiGraph,
    demands: list[dict],
    vectors: dict[tuple[int, int], dict[tuple[int, int], float]],
    scenario: dict,
) -> tuple[_BatchModel, dict[tuple[int, int], int], _XCols]:
    started = time.perf_counter()
    arcs = list(closure.edges)
    edges = list(graph.edges)
    n_arcs = len(arcs)
    n_dem = len(demands)
    slots = max(len(demand["v"]) for demand in demands)
    arc_index = {arc: i for i, arc in enumerate(arcs)}
    n_x = n_arcs * n_dem * slots
    n_z = len(edges)
    n_p = n_arcs * n_dem * max(slots - 1, 0)

    def x_col(arc: int, d: int, t: int) -> int:
        return (arc * n_dem + d) * slots + t

    def p_col(arc: int, d: int) -> int:
        return n_x + n_z + arc * n_dem + d

    problem = cplex.Cplex()
    problem.set_log_stream(None)
    problem.set_results_stream(None)
    problem.set_warning_stream(None)
    problem.objective.set_sense(problem.objective.sense.minimize)
    problem.parameters.timelimit.set(TIME_LIMIT)
    problem.parameters.mip.tolerances.mipgap.set(MIP_GAP)
    _add_vars(problem, n_x, 0.0, 1.0, "B")
    _add_vars(problem, n_z, 0.0, cplex.infinity, "C")
    _add_vars(problem, n_p, 0.0, 1.0, "C")
    for i in range(n_z):
        col = n_x + i
        problem.objective.set_quadratic_coefficients(col, col, 2.0)

    out_arcs = {node: [] for node in graph.nodes}
    in_arcs = {node: [] for node in graph.nodes}
    for i, (u, v) in enumerate(arcs):
        out_arcs[u].append(i)
        in_arcs[v].append(i)
    inds, vals, senses, rhs = [], [], [], []
    for d, demand in enumerate(demands):
        for t in range(slots):
            for q in graph.nodes:
                if q == demand["s"]:
                    balance = 1.0
                elif q == demand["t"]:
                    balance = -1.0
                else:
                    balance = 0.0
                row_i = [x_col(arc, d, t) for arc in out_arcs[q]]
                row_v = [1.0] * len(row_i)
                row_i += [x_col(arc, d, t) for arc in in_arcs[q]]
                row_v += [-1.0] * len(in_arcs[q])
                if not row_i and balance == 0.0:
                    continue
                inds.append(row_i)
                vals.append(row_v)
                senses.append("E")
                rhs.append(balance)
                if len(rhs) >= 5000:
                    _add_rows(problem, inds, vals, senses, rhs)
                    inds, vals, senses, rhs = [], [], [], []
    _add_rows(problem, inds, vals, senses, rhs)

    touched = {edge: [] for edge in edges}
    for (u, v), ratios in vectors.items():
        arc = arc_index.get((u, v))
        if arc is None:
            continue
        for edge, share in ratios.items():
            touched[edge].append((arc, share))
    for e, edge in enumerate(edges):
        for t in range(slots):
            row_i, row_v = [], []
            for arc, share in touched[edge]:
                for d in range(n_dem):
                    row_i.append(x_col(arc, d, t))
                    row_v.append(demands[d]["v"][t] * share)
            row_i.append(n_x + e)
            row_v.append(-1.0)
            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(row_i, row_v)], senses=["L"], rhs=[0.0]
            )

    inds, vals, senses, rhs = [], [], [], []
    for d in range(n_dem):
        for t in range(slots):
            inds.append([x_col(arc, d, t) for arc in range(n_arcs)])
            vals.append([1.0] * n_arcs)
            senses.append("L")
            rhs.append(float(scenario["max_segments"]))
    _add_rows(problem, inds, vals, senses, rhs)

    budget = {step["t"]: step["value"] for step in scenario["budget"]}
    for t in range(1, slots):
        inds, vals, senses, rhs = [], [], [], []
        total_i, total_v = [], []
        for arc in range(n_arcs):
            for d in range(n_dem):
                xt, xp, p = x_col(arc, d, t), x_col(arc, d, t - 1), p_col(arc, d)
                inds.append([p, xt])
                vals.append([1.0, -1.0])
                senses.append("L")
                rhs.append(0.0)
                inds.append([p, xp])
                vals.append([1.0, -1.0])
                senses.append("L")
                rhs.append(0.0)
                inds.append([p, xt, xp])
                vals.append([-1.0, 1.0, 1.0])
                senses.append("L")
                rhs.append(1.0)
                total_i.extend((xt, xp, p))
                total_v.extend((1.0, 1.0, -2.0))
                if len(rhs) >= 50_000:
                    _add_rows(problem, inds, vals, senses, rhs)
                    inds, vals, senses, rhs = [], [], [], []
        _add_rows(problem, inds, vals, senses, rhs)
        problem.linear_constraints.add(
            lin_expr=[cplex.SparsePair(total_i, total_v)],
            senses=["L"],
            rhs=[float(budget.get(t, 0))],
        )

    z = {edge: n_x + i for i, edge in enumerate(edges)}
    print(
        f"batch build {time.perf_counter() - started:.1f}s "
        f"vars={problem.variables.get_num()} constraints={problem.linear_constraints.get_num()}",
        flush=True,
    )
    return _BatchModel(problem), z, _XCols(arc_index, n_dem, slots)


def ip_model(
    graph: nx.DiGraph,
    closure: nx.DiGraph,
    demands: list[dict],
    vectors: dict[tuple[int, int], dict[tuple[int, int], float]],
    scenario: dict,
) -> tuple[Model, dict, dict]:
    if BATCH_BUILD:
        return ip_model_batch(graph, closure, demands, vectors, scenario)
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
