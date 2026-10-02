import itertools
from pathlib import Path

import matplotlib.pyplot as plt

from utils import (
    down_edges,
    load_demands,
    load_net,
    load_waypoints,
    loads,
    split_ratios,
)
from vis import draw_loads, draw_step, draw_topology

# prefix, same as `just check data/toy/toy`: loads <name>-net/tm/scenario/srpaths.json
INSTANCE = "data/toy/toy"
_base = Path(__file__).resolve().parents[1] / INSTANCE
NET_PATH = Path(f"{_base}-net.json")
TM_PATH = Path(f"{_base}-tm.json")
SCENARIO_PATH = Path(f"{_base}-scenario.json")
SRPATHS_PATH = Path(f"{_base}-srpaths.json")


def main() -> None:
    graph = load_net(NET_PATH)
    demands = load_demands(TM_PATH)
    names = {node: attrs["name"] for node, attrs in graph.nodes(data=True)}
    slots = max(len(demand["v"]) for demand in demands)
    waypoints = load_waypoints(SRPATHS_PATH)
    got_loads = loads(graph, demands, SCENARIO_PATH, waypoints)

    draw_topology(graph, names)
    for index, demand in enumerate(demands):
        label = rf"{names[demand['s']]} $\to$ {names[demand['t']]}"
        fig, axes = plt.subplots(1, slots, figsize=(10 * slots, 6), squeeze=False)
        fig.suptitle(label)
        for time, volume in enumerate(demand["v"]):
            active = set(graph.edges) - down_edges(graph, SCENARIO_PATH, time)
            live = graph.edge_subgraph(active).copy()
            via = waypoints.get((index, time), ())
            hops = [demand["s"], *via, demand["t"]]
            ratios: dict[tuple[int, int], float] = {}
            for src, dst in itertools.pairwise(hops):
                for edge, share in split_ratios(live, src, dst).items():
                    ratios[edge] = ratios.get(edge, 0.0) + share
            title = f"t = {time}  volume {volume:g}"
            if via:
                title += "  via " + ", ".join(names[node] for node in via)
            draw_step(
                axes[0][time],
                graph,
                names,
                active,
                ratios,
                title,
                source=demand["s"],
                sink=demand["t"],
                waypoints=via,
            )
        fig.tight_layout()
    plt.show()
    draw_loads(got_loads, "Sorted Arc Loads")


if __name__ == "__main__":
    main()
