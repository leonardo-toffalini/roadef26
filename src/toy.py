from pathlib import Path

import matplotlib.pyplot as plt

from utils import down_edges, load_demands, load_net, split_ratios
from vis import draw_step, draw_topology

DATA = Path(__file__).resolve().parents[1] / "data" / "toy"
NET_PATH = DATA / "toy-net.json"
TM_PATH = DATA / "toy-tm.json"
SCENARIO_PATH = DATA / "toy-scenario.json"

# fixed node positions so the toy net is recognizable
POS = {
    0: (1.0, 2.0),
    1: (3.0, 2.0),
    2: (0.0, 1.0),
    3: (4.0, 1.0),
    4: (2.0, 1.0),
    5: (3.0, 0.0),
    6: (1.0, 0.0),
}


def main() -> None:
    graph = load_net(NET_PATH)
    demands = load_demands(TM_PATH)
    names = {node: attrs["name"] for node, attrs in graph.nodes(data=True)}
    slots = max(len(demand["v"]) for demand in demands)

    expected = {
        (0, 1): 1,
        (1, 3): 0.5,
        (1, 4): 0.5,
        (3, 5): 0.75,
        (4, 3): 0.25,
        (4, 6): 0.25,
        (6, 5): 0.25,
    }
    assert split_ratios(graph, 0, 5) == expected

    draw_topology(graph, names, POS)
    for demand in demands:
        label = rf"{names[demand['s']]} $\to$ {names[demand['t']]}"
        fig, axes = plt.subplots(1, slots, figsize=(10 * slots, 6), squeeze=False)
        fig.suptitle(label)
        for time, volume in enumerate(demand["v"]):
            active = set(graph.edges) - down_edges(graph, SCENARIO_PATH, time)
            ratios = split_ratios(
                graph.edge_subgraph(active).copy(), demand["s"], demand["t"]
            )
            draw_step(
                axes[0][time],
                graph,
                names,
                active,
                ratios,
                f"t = {time}  volume {volume:g}",
                POS,
            )
        fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
