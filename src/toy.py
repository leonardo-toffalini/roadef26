import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

DATA = Path(__file__).resolve().parents[1] / "data" / "toy"
NET_PATH = DATA / "toy-net.json"
TM_PATH = DATA / "toy-tm.json"
SCENARIO_PATH = DATA / "toy-scenario.json"

# Fixed so the toy net is recognizable: A on the left, F/G on the right.
POS = {
    0: (1, 2),
    1: (3, 2),
    2: (0, 1),
    3: (4, 1),
    4: (2, 1),
    5: (3, 0),
    6: (1, 0),
}


def load_net(path: Path) -> nx.DiGraph:
    data = json.loads(path.read_text())
    return nx.node_link_graph(data, source="from", target="to", edges="links")


def load_demands(path: Path) -> list[dict]:
    return json.loads(path.read_text())["demands"]


def down_edges(graph: nx.DiGraph, path: Path, time: int) -> set[tuple[int, int]]:
    scenario = json.loads(path.read_text())
    link_ids = {
        link_id
        for step in scenario["interventions"]
        if step["t"] == time
        for link_id in step["links"]
    }
    return {
        (src, dst)
        for src, dst, attrs in graph.edges(data=True)
        if attrs["id"] in link_ids
    }


def split_ratios(
    graph: nx.DiGraph, source: int, target: int
) -> dict[tuple[int, int], float]:
    """ECMP fraction of a unit flow from source to target on shortest paths."""
    dist_s = nx.single_source_dijkstra_path_length(graph, source, weight="metric")
    dist_t = nx.single_source_dijkstra_path_length(
        graph.reverse(copy=False), target, weight="metric"
    )
    if target not in dist_s:
        return {}
    shortest = dist_s[target]
    dag = nx.DiGraph()
    for src, dst, attrs in graph.edges(data=True):
        if (
            src in dist_s
            and dst in dist_t
            and dist_s[src] + attrs["metric"] + dist_t[dst] == shortest
        ):
            dag.add_edge(src, dst)
    flow = {source: 1.0}
    ratios = {}
    for node in nx.topological_sort(dag):
        outs = list(dag.successors(node))
        if not outs or node not in flow:
            continue
        share = flow[node] / len(outs)
        for nxt in outs:
            ratios[(node, nxt)] = share
            flow[nxt] = flow.get(nxt, 0.0) + share
    return ratios


def draw_topology(graph: nx.DiGraph, names: dict[int, str]) -> None:
    edge_labels = {
        (src, dst): f"{attrs['metric']};{attrs['capacity']}"
        for src, dst, attrs in graph.edges(data=True)
    }
    arc = "arc3,rad=0.10"
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(graph, POS, node_color="#d6eaf8", node_size=500)
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
    plt.title("edge labels: metric;capacity")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def draw_step(ax, graph, names, active, ratios, title) -> None:
    arc = "arc3,rad=0.10"
    flow_arc = "arc3,rad=0.22"
    metric_labels = {
        (src, dst): f"{attrs['metric']};{attrs['capacity']}"
        for src, dst, attrs in graph.edges(data=True)
        if (src, dst) in active
    }
    ax.set_title(title)
    nx.draw_networkx_nodes(graph, POS, node_color="#d6eaf8", node_size=500, ax=ax)
    nx.draw_networkx_labels(graph, POS, labels=names, ax=ax)
    nx.draw_networkx_edges(
        graph,
        POS,
        edgelist=list(active),
        connectionstyle=arc,
        arrowsize=12,
        min_source_margin=15,
        min_target_margin=15,
        ax=ax,
    )
    down = [edge for edge in graph.edges if edge not in active]
    nx.draw_networkx_edges(
        graph,
        POS,
        edgelist=down,
        connectionstyle=arc,
        edge_color="#cccccc",
        arrowsize=12,
        min_source_margin=15,
        min_target_margin=15,
        ax=ax,
    )
    nx.draw_networkx_edge_labels(
        graph,
        POS,
        edge_labels=metric_labels,
        connectionstyle=arc,
        font_size=6,
        rotate=False,
        bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none"},
        ax=ax,
    )
    if ratios:
        nx.draw_networkx_edges(
            graph,
            POS,
            edgelist=list(ratios),
            connectionstyle=flow_arc,
            edge_color="#1f77b4",
            width=2,
            arrowsize=14,
            min_source_margin=15,
            min_target_margin=15,
            ax=ax,
        )
        nx.draw_networkx_edge_labels(
            graph,
            POS,
            edge_labels={edge: f"{share:g}" for edge, share in ratios.items()},
            connectionstyle=flow_arc,
            font_size=8,
            font_color="#1f77b4",
            rotate=False,
            bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none"},
            ax=ax,
        )
    ax.axis("off")


def main() -> None:
    graph = load_net(NET_PATH)
    demands = load_demands(TM_PATH)
    names = {node: attrs["name"] for node, attrs in graph.nodes(data=True)}
    slots = max(len(demand["v"]) for demand in demands)

    # Table 2, demand (v0, v5) at t = 0, no links down.
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

    draw_topology(graph, names)
    for demand in demands:
        label = f"{names[demand['s']]}→{names[demand['t']]}"
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
            )
        fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
