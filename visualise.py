import igraph
import matplotlib.pyplot as pyplot

from cpp.graphutil import read_graph_from_file


def make_graph(weighted_edges):
    edges = [edge[0:2] for edge in weighted_edges]
    weights = [edge[2] for edge in weighted_edges]

    return igraph.Graph(
        edges=edges,
        edge_attrs={'weight': weights}
    )


def main():
    # list(range(graph.vcount()))
    graph_file = "graph_20.dat"
    graph = read_graph_from_file(graph_file)
    edges_weights = [graph.es[i]['weight'] for i in range(graph.ecount())]
    visual_style = {
        "vertex_label": list(range(graph.vcount())),
        "vertex_label_color": "white",
        "vertex_size": 0.3,
        "edge_label": edges_weights,
        "edge_label_color": "blue"
    }
    layout = graph.layout()
    _, ax = pyplot.subplots()
    igraph.drawing.plot(graph, layout=layout, target=ax, **visual_style)
    pyplot.show()


if __name__ == "__main__":
    main()
