"""Local search from ls.py, with closure shortest paths in place of the waypoint scan.

When a demand uses one of the hottest links, its replacement route is a
shortest path in the reachability closure of that time slot, of length at
most ``MAX_PATH_LENGTH`` closure edges. Length 2 is at most one waypoint.
The closure edge uv costs the linearized L^16 of sending the demand from u
to v by ECMP, priced at the load with this demand removed:

    sum over links of  16 * base^15 * (unit flow / capacity)

Volume scales every edge by the same positive constant, so it does not change
the path. The path is accepted only when the exact score used by ls.py
improves and the change stays inside the budget.
"""

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ls import (
    DATA,
    EPS,
    HOT,
    PER_ARC,
    TIME_LIMIT,
    change,
    load,
    pow16,
    soft,
    unit_flows,
    write,
)

INSTANCES_A = [f"setA/setA-{i:02d}" for i in range(1, 21)]
INSTANCES_B = [f"setB/setB-{i:02d}" for i in range(1, 13)]
INSTANCES_C = [f"setC/setC-{i:02d}" for i in range(1, 13)]
INSTANCES_X = [f"setX/setX-{i:02d}" for i in range(1, 13)]
# INSTANCES = INSTANCES_A + INSTANCES_B  # + INSTANCES_C + INSTANCES_X
INSTANCES = INSTANCES_C
OUT = Path(__file__).resolve().parents[1] / "out" / "ls_closure"
# closure edges on one route. 2 is s-w-t, so at most one waypoint.
# also never exceeds the instance's max_segments.
MAX_PATH_LENGTH = 2


def closure_via(flow, base: np.ndarray, s: int, dest: int, hops: int) -> tuple:
    """Waypoints of a cheapest s-t path of at most ``hops`` closure edges.

    Fewer hops wins a tie, so a zero-cost detour is not added. An empty tuple
    is the direct route. Returns () when s cannot reach dest within the limit.
    """
    if hops < 1 or s == dest:
        return ()
    ptr, cols, vals, reach = flow
    n = int(reach.shape[0])
    price = _price(base)
    if hops <= 2:
        # only s -> w and w -> dest can be on the path: price 2n edges, not all n^2
        w = np.arange(n)
        out = np.where(reach[s], _pair_cost(flow, price, s * n + w), np.inf)
        into = np.where(reach[:, dest], _pair_cost(flow, price, w * n + dest), np.inf)
        two = out + into
        two[[s, dest]] = np.inf
        k = int(np.argmin(two))
        # fewer hops wins a tie
        return (k,) if hops == 2 and two[k] < out[dest] - 1e-9 else ()
    lengths = np.diff(ptr)
    row = np.repeat(np.arange(n * n), lengths)
    cost = np.bincount(row, weights=vals * price[cols], minlength=n * n)
    cost = cost.reshape(n, n)
    cost[~reach] = np.inf
    np.fill_diagonal(cost, np.inf)
    # s is only the start and dest is only the end, same as a waypoint list
    cost[:, s] = np.inf
    cost[dest, :] = np.inf

    dist = np.full((hops + 1, n), np.inf)
    prev = np.full((hops + 1, n), -1, np.int32)
    dist[0, s] = 0.0
    for k in range(1, hops + 1):
        at = dist[k - 1].copy()
        at[dest] = np.inf
        choice = at[:, None] + cost
        prev[k] = np.argmin(choice, axis=0).astype(np.int32)
        dist[k] = choice[prev[k], np.arange(n)]
        prev[k, ~np.isfinite(dist[k])] = -1

    finite = np.flatnonzero(np.isfinite(dist[1:, dest]))
    if finite.size == 0:
        return ()
    best = dist[1:, dest][finite].min()
    k = int(finite[dist[1:, dest][finite] <= best + 1e-9][0]) + 1
    nodes = [dest]
    v = dest
    for step in range(k, 0, -1):
        v = int(prev[step, v])
        if v < 0 or v in nodes:
            return ()
        nodes.append(v)
    nodes.reverse()
    if nodes[0] != s or len(nodes) != k + 1:
        return ()
    return tuple(nodes[1:-1])


def _pair_cost(flow, price: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Priced cost of each closure edge in pairs, keyed u * n + v like the CSR rows."""
    ptr, cols, vals, _ = flow
    starts = ptr[pairs]
    lens = ptr[pairs + 1] - starts
    flat = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
    owner = np.repeat(np.arange(len(pairs)), lens)
    return np.bincount(
        owner, weights=vals[flat] * price[cols[flat]], minlength=len(pairs)
    )


def _price(base: np.ndarray) -> np.ndarray:
    """Derivative of load^16 at the current load. Zero where the link is empty."""
    price = np.zeros(base.shape)
    hot = base > 0
    price[hot] = 16 * pow16(base[hot]) / base[hot]
    return price


def search(
    n,
    links,
    demands,
    scenario,
    time_limit=TIME_LIMIT,
    log=True,
    max_path_length=MAX_PATH_LENGTH,
):
    slots = len(demands[0]["v"])
    n_arcs, n_dem = len(links), len(demands)
    down = {t: set() for t in range(slots)}
    for step in scenario["interventions"]:
        down[step["t"]] |= set(step["links"])
    budget = {step["t"]: step["value"] for step in scenario["budget"]}
    # one closure edge is one segment. max_path_length 2 allows one waypoint.
    hops = min(scenario["max_segments"], max_path_length)
    t0 = time.perf_counter()
    # slots are independent, so build them in parallel processes
    with ProcessPoolExecutor() as pool:
        flows = list(pool.map(unit_flows, [n] * slots, [links] * slots, [down[t] for t in range(slots)]))
    setup = time.perf_counter() - t0
    vol = np.array([demand["v"] for demand in demands], float)
    src = [demand["s"] for demand in demands]
    dst = [demand["t"] for demand in demands]

    def entries(t: int, d: int, cands: list[tuple]):
        """Utilization each candidate route of d adds at slot t, as sparse
        (candidate, arc, value) sorted by candidate, and whether it routes."""
        ptr, cols, vals, reach = flows[t]
        ok = np.ones(len(cands), bool)
        owner, pairs = [], []
        for k, via in enumerate(cands):
            hop = (src[d], *via, dst[d])
            for a, b in zip(hop, hop[1:]):
                ok[k] &= reach[a, b]
                owner.append(k)
                pairs.append(a * n + b)
        pairs = np.array(pairs)
        starts = ptr[pairs]
        lens = ptr[pairs + 1] - starts
        flat = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
        # segments of one route can share arcs
        key, inverse = np.unique(
            np.repeat(owner, lens) * n_arcs + cols[flat], return_inverse=True
        )
        val = np.bincount(inverse, weights=vals[flat] * vol[d, t], minlength=len(key))
        return key // n_arcs, key % n_arcs, val, ok

    def dense(owner, col, val, k: int) -> np.ndarray:
        row = np.zeros(n_arcs)
        mask = owner == k
        row[col[mask]] = val[mask]
        return row

    routes: list[list[tuple]] = [[() for _ in range(slots)] for _ in range(n_dem)]
    contrib = np.zeros((slots, n_dem, n_arcs))
    for t in range(slots):
        for d in range(n_dem):
            owner, col, val, ok = entries(t, d, [()])
            assert ok[0], (d, t)
            contrib[t, d, col] = val
    util = contrib.sum(1)
    spent = {t: 0 for t in range(1, slots)}

    def state() -> tuple[np.ndarray, np.ndarray]:
        return util.max(1), soft(util)

    def better(g: float, s: float, g0: float, s0: float) -> bool:
        return g < g0 - EPS or (g <= g0 + EPS and s < s0 * (1 - EPS))

    def evaluate(d: int, cands: list[tuple]):
        """Best feasible move of d among cands, for one slot or all slots at once."""
        peak, mass = state()
        rows, gm, sm, ok = [], [], [], []
        for t in range(slots):
            owner, col, val, fine = entries(t, d, cands)
            base = util[t] - contrib[t, d]
            new = base[col] + val
            # loads only grow on the route's arcs, so the rest keeps base's max
            g = np.full(len(cands), base.max())
            present, first = np.unique(owner, return_index=True)
            g[present] = np.maximum(g[present], np.maximum.reduceat(new, first))
            gain = np.bincount(
                owner, weights=pow16(new) - pow16(base[col]), minlength=len(cands)
            )
            rows.append((owner, col, val))
            gm.append(g)
            sm.append(soft(base) + gain)
            ok.append(fine)
        best = None
        g0, s0 = peak.max(), mass.sum()
        # one slot changes, or every slot takes the same route
        for ts in [[t] for t in range(slots)] + [list(range(slots))]:
            rest = [t for t in range(slots) if t not in ts]
            g = np.max([gm[t] for t in ts], axis=0)
            if rest:
                g = np.maximum(g, peak[rest].max())
            s = np.sum([sm[t] for t in ts], axis=0) + mass[rest].sum()
            feasible = np.logical_and.reduce([ok[t] for t in ts])
            for k in np.lexsort((s, g)):
                if not feasible[k]:
                    continue
                if not better(g[k], s[k], g0, s0):
                    break
                new_routes = list(routes[d])
                for t in ts:
                    new_routes[t] = cands[k]
                cost = {
                    t: spent[t]
                    - change(routes[d][t - 1], routes[d][t], src[d], dst[d])
                    + change(new_routes[t - 1], new_routes[t], src[d], dst[d])
                    for t in spent
                }
                if any(cost[t] > budget[t] for t in cost):
                    continue
                if best is None or better(g[k], s[k], best[0], best[1]):
                    best = (
                        g[k],
                        s[k],
                        new_routes,
                        {t: dense(*rows[t], k) for t in ts},
                        cost,
                    )
                break
        return best

    def best_move(d: int, t: int):
        via = closure_via(flows[t], util[t] - contrib[t, d], src[d], dst[d], hops)
        if via == tuple(routes[d][t]):
            return None
        return evaluate(d, [via])

    started = time.perf_counter()
    first = state()[0]
    moves = 0
    skip: set[tuple[int, int]] = set()
    cleared = False
    while time.perf_counter() - started < time_limit:
        moved = False
        for flat in np.argsort(util, axis=None)[::-1][:HOT]:
            t, a = divmod(int(flat), n_arcs)
            col = contrib[t, :, a]
            for d in np.argsort(col)[::-1][:PER_ARC]:
                if time.perf_counter() - started >= time_limit:
                    break
                d = int(d)
                if col[d] <= 0:
                    break
                if (d, t) in skip:
                    continue
                move = best_move(d, t)
                if move is None:
                    skip.add((d, t))
                    continue
                _, _, new_routes, rows, cost = move
                for slot, row in rows.items():
                    util[slot] += row - contrib[slot, d]
                    contrib[slot, d] = row
                routes[d] = new_routes
                spent.update(cost)
                moves += 1
                moved = True
                break
            if moved or time.perf_counter() - started >= time_limit:
                break
        if moved:
            cleared = False
            continue
        # don't-look bits: retry skipped demands once before stopping
        if cleared:
            break
        skip.clear()
        cleared = True
    if log:
        print(
            f"  {moves} moves, mlu {np.round(first, 4).tolist()} -> "
            f"{np.round(state()[0], 4).tolist()}, spent {spent}, "
            f"unit flows {setup:.1f}s, search {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return routes, util, spent


def _pack(n: int, triples: list[tuple[int, int, int, float]]):
    buckets: list[list[tuple[int, float]]] = [[] for _ in range(n * n)]
    for u, v, arc, val in triples:
        buckets[u * n + v].append((arc, val))
    ptr = [0]
    cols: list[int] = []
    vals: list[float] = []
    for bucket in buckets:
        cols.extend(arc for arc, _ in bucket)
        vals.extend(val for _, val in bucket)
        ptr.append(len(cols))
    reach = np.zeros((n, n), bool)
    for u, v, _, _ in triples:
        reach[u, v] = True
    reach[np.arange(n), np.arange(n)] = True
    return np.array(ptr), np.array(cols), np.array(vals, float), reach


def _check() -> None:
    from utils import load_demands, load_net, loads

    # direct arc 2 is priced, so the cheap route detours through node 1
    flow = _pack(3, [(0, 1, 0, 1.0), (1, 2, 1, 1.0), (0, 2, 2, 1.0)])
    base = np.array([0.0, 0.0, 1.0])
    assert closure_via(flow, base, 0, 2, 1) == ()
    assert closure_via(flow, base, 0, 2, 2) == (1,)
    # every closure edge is free, so the fewest hops is the direct route
    assert closure_via(flow, np.zeros(3), 0, 2, 3) == ()

    name = "setA/setA-01"
    n, links, demands, scenario = load(name)
    routes, util, spent = search(
        n, links, demands, scenario, time_limit=5, log=False, max_path_length=2
    )
    for route in routes:
        for via in route:
            assert len(via) <= 1
            assert len(set(via)) == len(via)
    base = DATA / name
    waypoints = {
        (d, t): list(via)
        for d, route in enumerate(routes)
        for t, via in enumerate(route)
    }
    want = loads(
        load_net(Path(f"{base}-net.json")),
        load_demands(Path(f"{base}-tm.json")),
        Path(f"{base}-scenario.json"),
        waypoints,
    )
    for t in range(util.shape[0]):
        for i, link in enumerate(links):
            expected = want.get((link["from"], link["to"], t), 0.0)
            assert abs(util[t, i] - expected) < 1e-9, (t, i, util[t, i], expected)
    for t, cost in spent.items():
        assert cost == sum(
            change(r[t - 1], r[t], dm["s"], dm["t"]) for r, dm in zip(routes, demands)
        )
        assert cost <= {s["t"]: s["value"] for s in scenario["budget"]}[t]


if __name__ == "__main__":
    _check()
    print("self-check ok", flush=True)
    for name in INSTANCES:
        print(name, flush=True)
        n, links, demands, scenario = load(name)
        routes, _, _ = search(n, links, demands, scenario)
        write(routes, OUT / f"{Path(name).name}-srpaths.json")
