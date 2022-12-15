import sys

import igraph

from cpp.cpp2 import solve_cpp
from cpp.graphutil import read_graph_from_file
from cpp.cpp import auto_solve_cpp
import matplotlib.pyplot as pyplot
def main():
    # graph_file = None
    # if len(sys.argv) > 1:
    #     graph_file = sys.argv[1]
    # else:
    #     print("usage: python main.py <graph_file>")
    #     exit(-1)

    # graph = read_graph_from_file(graph_file)
    graph = read_graph_from_file("./graphs/graph_small.dat")
    # path_vertices, euler_graph, graph_type, cost = auto_solve_cpp(graph)
    solution = solve_cpp(graph)
    print(solution)

    visual_style = {
        "vertex_label_color": "blue",
        "vertex_label": range(graph.vcount()),
        # "vertex_color": "blue",
        "vertex_size": 0.3
    }
    layout = graph.layout()
    fig, ax = pyplot.subplots()
    igraph.drawing.plot(graph, layout=layout, target=ax, **visual_style)
    # igraph.drawing.plot(graph.linegraph(), layout=layout, target=axs[1], **visual_style)
    pyplot.show()
    # print(f"The input graph is {str(graph_type)} and the transformation cost is {cost}")
    # print(f"The Postman's paths is: {path_vertices}")
    # optionally visualize the euler_graph


if __name__ == "__main__":
    main()
