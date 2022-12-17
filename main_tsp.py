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
    path_vertices, cost = solve(graph, matrix, ga_instance)
    ga_instance.plot_fitness()

    print(f"The best solution fitness is {cost}")
    print(f"The Postman's paths is: {path_vertices}")
    print(f"which is {len(path_vertices) + 1} vertices")
    u = set(path_vertices)
    print(f"total retraversals: {len(path_vertices) - len(u)} ")


if __name__ == "__main__":
    main()
