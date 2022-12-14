import sys

from cpp.graphutil import read_graph_from_file
from cpp.ga.euler import solve


def main():
    graph_file = None
    if len(sys.argv) > 1:
        graph_file = sys.argv[1]
    else:
        print("usage: python main_euler.py <graph_file>")
        exit(-1)

    graph = read_graph_from_file(graph_file)
    path_vertices, euler_graph, graph_type, cost = solve(graph)

    print(f"The input graph is {str(graph_type)} and the transformation cost is {cost}")
    print(f"The Postman's paths is: {path_vertices}")
    print(f"which is {len(path_vertices)} vertices")


if __name__ == "__main__":
    main()
