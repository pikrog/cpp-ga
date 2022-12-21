import sys

from cpp.ga.tsp import solve, create_matrix, create_template_ga_instance
from cpp.ga.util import abs_pmx_crossover, abs_pmx_insert
from cpp.graphutil import read_graph_from_file


def main():
    graph_file = None
    if len(sys.argv) > 1:
        graph_file = sys.argv[1]
    else:
        print("usage: python main_tsp.py <graph_file>")
        exit(-1)

    # a = abs_pmx_insert([8, 4, 7, 3, 6, 2, 5, 1, 9, 0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 8)
    # print(a)
    # assert(a == [0, 7, 4, 3, 6, 2, 5, 1, 8, 9])
    # return

    graph = read_graph_from_file(graph_file)
    matrix = create_matrix(graph)

    while True:
        ga_instance = create_template_ga_instance(
            graph, matrix,
            num_generations=2000, stop_criteria=["saturate_500"],
            crossover_probability=0.8,
            mutation_probability=0.025,
            population_size=100, num_parents_mating=75,
            keep_elitism=0,
            keep_parents=2,
            parent_selection_type="sss",    # sss & tournament - so far the best
        )
        vertex_path, cost, edge_path = solve(graph, matrix, ga_instance)

        print(f"The best solution path cost is {cost}")
        print(f"The Postman's paths is: {vertex_path}")
        print(f"which contains {len(vertex_path)} vertices")

        ga_instance.plot_fitness()


if __name__ == "__main__":
    main()
