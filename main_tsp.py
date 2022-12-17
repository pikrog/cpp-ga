import sys

from cpp.graphutil import read_graph_from_file
from cpp.ga.tsp import solve, create_matrix, create_template_ga_instance


def main():
    graph_file = None
    if len(sys.argv) > 1:
        graph_file = sys.argv[1]
    else:
        print("usage: python main.py <graph_file>")
        exit(-1)

    graph = read_graph_from_file(graph_file)
    matrix = create_matrix(graph)
    ga_instance = create_template_ga_instance(graph, matrix)
    vertex_path, cost, _, edge_path = solve(graph, matrix, ga_instance)
    ga_instance.plot_fitness()

    print(f"The best solution fitness is {cost}")
    print(f"The Postman's paths is: {vertex_path}")
    print(f"which contains {len(vertex_path)} vertices")
    print(edge_path)
    # u = set(vertex_path)
    # print(f"total retraversals: {len(vertex_path) - len(u)} ")


if __name__ == "__main__":
    main()
