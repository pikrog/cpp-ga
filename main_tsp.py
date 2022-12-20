import sys

from cpp.ga.tsp import solve, create_matrix, create_template_ga_instance
from cpp.graphutil import read_graph_from_file


def main():
    graph_file = None
    if len(sys.argv) > 1:
        graph_file = sys.argv[1]
    else:
        print("usage: python main_euler.py <graph_file>")
        exit(-1)

    graph = read_graph_from_file(graph_file)
    matrix = create_matrix(graph)

    ga_instance = create_template_ga_instance(graph, matrix)
    vertex_path, cost, edge_path = solve(graph, matrix, ga_instance)

    print(f"The best solution path cost is {cost}")
    print(f"The Postman's paths is: {vertex_path}")
    print(f"which contains {len(vertex_path)} vertices")
    print(edge_path)


if __name__ == "__main__":
    main()
