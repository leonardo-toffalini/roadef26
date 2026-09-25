import matplotlib.pyplot as plt
import networkx as nx


def draw_topology(
    graph: nx.DiGraph, names: dict[int, str], positions: dict[int, tuple[float, float]]
) -> None:
    edge_labels = {
        (src, dst): f"{attrs['metric']};{attrs['capacity']}"
        for src, dst, attrs in graph.edges(data=True)
    }
    arc = "arc3,rad=0.10"
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(graph, positions, node_color="#d6eaf8", node_size=500)
    nx.draw_networkx_labels(graph, positions, labels=names)
    nx.draw_networkx_edges(
        graph,
        positions,
        connectionstyle=arc,
        arrowsize=12,
        min_source_margin=15,
        min_target_margin=15,
    )
    nx.draw_networkx_edge_labels(
        graph,
        positions,
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


def draw_step(
    ax,
    graph: nx.DiGraph,
    names: dict[int, str],
    active: set[tuple[int, int]],
    ratios: dict[tuple[int, int], float],
    title: str,
    positions: dict[int, tuple[float, float]],
) -> None:
    arc = "arc3,rad=0.10"
    flow_arc = "arc3,rad=0.22"
    metric_labels = {
        (src, dst): f"{attrs['metric']};{attrs['capacity']}"
        for src, dst, attrs in graph.edges(data=True)
        if (src, dst) in active
    }
    ax.set_title(title)
    nx.draw_networkx_nodes(graph, positions, node_color="#d6eaf8", node_size=500, ax=ax)
    nx.draw_networkx_labels(graph, positions, labels=names, ax=ax)
    nx.draw_networkx_edges(
        graph,
        positions,
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
        positions,
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
        positions,
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
            positions,
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
            positions,
            edge_labels={edge: f"{share:g}" for edge, share in ratios.items()},
            connectionstyle=flow_arc,
            font_size=8,
            font_color="#1f77b4",
            rotate=False,
            bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none"},
            ax=ax,
        )
    ax.axis("off")
