import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

NET_PATH = Path(__file__).resolve().parents[1] / "data" / "toy" / "toy-net.json"

# Fixed so the toy net is recognizable: A on the left, F/G on the right.
POS = {
    0: (0, 1),
    1: (1, 2),
    2: (1, 0),
    3: (2, 2),
    4: (2, 1),
    5: (3, 2),
    6: (3, 0),
}


def load_net(path: Path) -> nx.DiGraph:
    data = json.loads(path.read_text())
    return nx.node_link_graph(data, source="from", target="to", edges="links")


def main() -> None:
    graph = load_net(NET_PATH)
    names = {node: attrs["name"] for node, attrs in graph.nodes(data=True)}
    edge_labels = {
        (src, dst): f"{attrs['metric']}/{attrs['capacity']}"
        for src, dst, attrs in graph.edges(data=True)
    }
    arc = "arc3,rad=0.28"

    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(graph, POS, node_color="#d6eaf8", node_size=700)
    nx.draw_networkx_labels(graph, POS, labels=names)
    nx.draw_networkx_edges(
        graph,
        POS,
        connectionstyle=arc,
        arrowsize=12,
        min_source_margin=15,
        min_target_margin=15,
    )
    nx.draw_networkx_edge_labels(
        graph,
        POS,
        edge_labels=edge_labels,
        connectionstyle=arc,
        font_size=7,
        rotate=False,
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none"},
    )
    plt.title("edge labels: metric / capacity")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
