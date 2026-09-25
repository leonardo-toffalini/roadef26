import json
from pathlib import Path

import networkx as nx


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
    """
    Calculates ECMP fraction of a unit flow from source to target on shortest paths,
    as described in Section 2 in problem_setting.pdf.
    """
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
