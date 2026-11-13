import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

# Local search over waypoints, scored like the checker: utilization is
# traffic / capacity per (arc, slot), with that slot's failed links removed.
# A move is kept when it lowers (max utilization, sum of utilization^16)
# lexicographically. The second term pushes down the next-highest arcs once
# the max is stuck, which is what the lexmin objective asks for.
INSTANCES = [f"setB/setB-{i:02d}" for i in range(1, 13)]
DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "out" / "ls"
# seconds per instance
TIME_LIMIT = 120
# (arc, slot) entries scanned per step, hottest first
HOT = 30
# demands tried per hot arc, largest contribution first
PER_ARC = 50
# best single waypoints extended to two-waypoint routes
TOP_FIRST = 4
EPS = 1e-9


def load(name: str):
    base = DATA / name
    net = json.loads(Path(f"{base}-net.json").read_text())
    demands = json.loads(Path(f"{base}-tm.json").read_text())["demands"]
    scenario = json.loads(Path(f"{base}-scenario.json").read_text())
    n = len(net["nodes"])
    assert sorted(node["id"] for node in net["nodes"]) == list(range(n))
    return n, net["links"], demands, scenario


def unit_flows(n: int, links: list[dict], down: set[int]):
    """ECMP unit flow of every ordered pair, as CSR rows indexed u * n + v.

    Values are divided by capacity, so they add up to utilization. reach[u, v]
    is False when v cannot be reached from u.
    """
    out: list[list[int]] = [[] for _ in range(n)]
    rev = nx.DiGraph()
    rev.add_nodes_from(range(n))
    for i, link in enumerate(links):
        if link["id"] not in down:
            out[link["from"]].append(i)
            rev.add_edge(link["to"], link["from"], metric=link["metric"])
    cap = np.array([link["capacity"] for link in links], float)
    reach = np.zeros((n, n), bool)
    keys, cols, vals = [], [], []
    for v in range(n):
        dist = nx.single_source_dijkstra_path_length(rev, v, weight="metric")
        sources = np.fromiter(dist, int)
        reach[sources, v] = True
        # at[x, s]: flow from source s arriving at x
        at = np.zeros((n, n))
        at[sources, sources] = 1.0
        edges, shares = [], []
        # metrics are positive, so every predecessor is farther from v
        for x in sorted(dist, key=dist.get, reverse=True):
            if x == v:
                continue
            # summed metrics miss equality by a few ulps on longer paths
            tol = 1e-8 * max(1.0, dist[x])
            nxt = [
                i
                for i in out[x]
                if links[i]["to"] in dist
                and abs(dist[x] - links[i]["metric"] - dist[links[i]["to"]]) <= tol
            ]
            share = at[x] / len(nxt)
            for i in nxt:
                at[links[i]["to"]] += share
                edges.append(i)
                shares.append(share)
        if not edges:
            continue
        flow = np.array(shares)
        row, source = np.nonzero(flow)
        edge = np.array(edges)[row]
        keys.append(source * n + v)
        cols.append(edge)
        vals.append(flow[row, source] / cap[edge])
    key = np.concatenate(keys)
    order = np.argsort(key, kind="stable")
    ptr = np.zeros(n * n + 1, np.int64)
    ptr[1:] = np.cumsum(np.bincount(key, minlength=n * n))
    return ptr, np.concatenate(cols)[order], np.concatenate(vals)[order], reach


def change(a: tuple, b: tuple, s: int, t: int) -> int:
    """The checker's budget cost: segments in one route but not the other."""
    p, q = (s, *a, t), (s, *b, t)
    return len(set(zip(p, p[1:])) ^ set(zip(q, q[1:])))


def pow16(x: np.ndarray) -> np.ndarray:
    for _ in range(4):
        x = x * x
    return x


def soft(x: np.ndarray) -> np.ndarray:
    """Sum of x^16 along the last axis."""
    return pow16(x).sum(-1)


def search(n, links, demands, scenario, time_limit=TIME_LIMIT, log=True):
    slots = len(demands[0]["v"])
    n_arcs, n_dem = len(links), len(demands)
    down = {t: set() for t in range(slots)}
    for step in scenario["interventions"]:
        down[step["t"]] |= set(step["links"])
    budget = {step["t"]: step["value"] for step in scenario["budget"]}
    max_via = scenario["max_segments"] - 1
    t0 = time.perf_counter()
    flows = [unit_flows(n, links, down[t]) for t in range(slots)]
    if log:
        print(f"  unit flows {time.perf_counter() - t0:.1f}s", flush=True)
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
            hops = (src[d], *via, dst[d])
            for a, b in zip(hops, hops[1:]):
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
        return best, g, s

    def best_move(d: int):
        others = [w for w in range(n) if w not in (src[d], dst[d])]
        cands = [()] + [(w,) for w in others] if max_via >= 1 else [()]
        best, g, s = evaluate(d, cands)
        if best is not None or max_via < 2:
            return best
        top = [cands[k][0] for k in np.lexsort((s, g)) if cands[k]][:TOP_FIRST]
        pairs = [(a, b) for a in top for b in others if b != a]
        return evaluate(d, pairs)[0]

    started = time.perf_counter()
    first = state()[0]
    moves = 0
    skip: set[int] = set()
    cleared = False
    while time.perf_counter() - started < time_limit:
        moved = False
        for flat in np.argsort(util, axis=None)[::-1][:HOT]:
            t, a = divmod(int(flat), n_arcs)
            col = contrib[t, :, a]
            for d in np.argsort(col)[::-1][:PER_ARC]:
                d = int(d)
                if col[d] <= 0:
                    break
                if d in skip:
                    continue
                move = best_move(d)
                if move is None:
                    skip.add(d)
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
            if moved:
                break
        if moved:
            cleared = False
            if log and moves % 25 == 0:
                print(
                    f"  move {moves} mlu {np.round(state()[0], 4).tolist()} "
                    f"spent {spent} {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
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
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return routes, util, spent


def write(routes: list[list[tuple]], path: Path) -> None:
    rows = [
        {"d": d, "t": t, "w": list(via)}
        for d, route in enumerate(routes)
        for t, via in enumerate(route)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"srpaths": rows}, indent=2) + "\n")


def _check() -> None:
    from utils import load_demands, load_net, loads

    name = "setA/setA-01"
    n, links, demands, scenario = load(name)
    routes, util, spent = search(n, links, demands, scenario, time_limit=5, log=False)
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
