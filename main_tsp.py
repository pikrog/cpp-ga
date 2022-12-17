import sys

from cpp.graphutil import read_graph_from_file
from cpp.ga.tsp import solve


def main():
    graph_file = None
    if len(sys.argv) > 1:
        graph_file = sys.argv[1]
    else:
        print("usage: python main.py <graph_file>")
        exit(-1)

    graph = read_graph_from_file(graph_file)
    path_vertices, cost = solve(graph)

    print(f"The best solution fitness is {cost}")
    print(f"The Postman's paths is: {path_vertices}")
    print(f"which is {len(path_vertices)} vertices")


if __name__ == "__main__":
    main()
