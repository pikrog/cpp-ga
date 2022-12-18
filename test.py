import sys
import numpy

from cpp.graphutil import read_graph_from_file, GraphType
from cpp.ga.euler import OddVerticesPathMatrix, create_template_ga_instance, find_euler_transform


def test():
    graph_file = None
    if len(sys.argv) > 1:
        graph_file = sys.argv[1]
    else:
        print("usage: python test.py <graph_file>")
        exit(-1)

    graph = read_graph_from_file(graph_file)
    assert(GraphType.of(graph) is GraphType.generic)
    assert(graph.is_connected())

    matrix = OddVerticesPathMatrix(graph)

    test_repeats = 10
    all_test_results = []

    population_size = 30
    num_generations = 100
    crossover_probability = 0.9
    mutation_probabilities = numpy.arange(start=0, stop=0.2, step=0.025)
    for mutation_probability in mutation_probabilities:

        print(f"testing mutation_probability={mutation_probability}")

        test_results = []
        for i in range(test_repeats):

            print(f"{i+1}, ", end='')

            ga_instance = create_template_ga_instance(
                matrix=matrix,
                population_size=population_size,
                num_generations=num_generations,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
            )
            _, cost, _ = find_euler_transform(matrix, ga_instance)
            test_results.append(cost)
        all_test_results.append(test_results)

        print("completed")

    all_tests_results_array = numpy.array(all_test_results)
    global_min_cost = numpy.min(all_tests_results_array)

    indicators = numpy.sqrt(
        numpy.sum(
            all_tests_results_array**2 - global_min_cost**2, axis=1
        )/test_repeats
    )
    best_indicator_index = numpy.argmin(indicators)
    best_indicator = indicators[best_indicator_index]
    best_mutation_probability = mutation_probabilities[best_indicator_index]

    print(f"global_min_cost={global_min_cost}")
    print(f"best_indicator={best_indicator}, best_mutation_probability={best_mutation_probability}")

    print(f"indicators: {indicators}")
    print(f"all test results: {all_test_results}")


if __name__ == "__main__":
    test()
