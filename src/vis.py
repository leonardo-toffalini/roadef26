import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams["mathtext.fontset"] = "cm"


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


def draw_loads(loads: dict[tuple[int, int, int], float], title: str) -> None:
    """Non-zero loads sorted decreasing, as in Figure 4 of problem_setting.pdf."""
    ranked = sorted(
        ((load, src, dst, time) for (src, dst, time), load in loads.items() if load > 0),
        reverse=True,
    )
    labels = [rf"$(a_{{{src},{dst}}}, {time})$" for _, src, dst, time in ranked]
    heights = [load for load, *_ in ranked]
    top = max(heights, default=1.0)
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(labels)), 5))
    ax.bar(range(len(heights)), heights, color="#7F7FFF")
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_ylim(0, 1.0 if top <= 1 else top * 1.05)
    ax.set_xlabel("Arc and Time Slot")
    ax.set_ylabel(r"Load $\lambda(a, t)$")
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(True)
    fig.tight_layout()
    plt.show()