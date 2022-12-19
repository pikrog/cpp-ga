import sys

from cpp.graphutil import read_graph_from_file
from cpp.ga.tsp import solve, create_matrix, create_template_ga_instance, _interpret_ga_solution, \
    _generate_random_solutions


def main():
    graph_file = None
    if len(sys.argv) > 1:
        graph_file = sys.argv[1]
    else:
        print("usage: python main.py <graph_file>")
        exit(-1)

    graph = read_graph_from_file(graph_file)
    matrix = create_matrix(graph)
    # return
    # print(_interpret_ga_solution(graph, matrix,
    #                              [[0, 3, 5, 9, 11, 10, 16, 14, 12, 13, 15, 17, 18, 19, 21,
    #                                22, 2, 1, 4, 6, 27, 20, 26, 7, 8, 24, 23, 25, 28, 25]]))
    # print(_interpret_ga_solution(graph, matrix,
    #                              [[1, 4, 6, 10, 12, 11, 17, 15, 13, 14, 16, 18, 19, 20, 22,
    #                                23, 3, 2, 5, 7, 28, 21, 27, 8, 9, 25, 24, 26, 29]]))
    # return
    while True:
        ga_instance = create_template_ga_instance(graph, matrix)
        vertex_path, cost, edge_path = solve(graph, matrix, ga_instance)
        #ga_instance.plot_fitness()

        print(f"The best solution path cost is {cost}")
        print(f"The Postman's paths is: {vertex_path}")
        print(f"which contains {len(vertex_path)} vertices")
        print(edge_path)
        if cost == 232:
            break
    # u = set(vertex_path)
    # print(f"total retraversals: {len(vertex_path) - len(u)} ")


if __name__ == "__main__":
    main()
