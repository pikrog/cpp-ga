import numpy
from time import time

from cpp.graphutil import read_graph_from_file, GraphType
from cpp.ga.tsp import create_matrix, create_template_ga_instance, solve

population_sizes = numpy.arange(start=30, stop=501, step=10)
nums_generations = numpy.arange(start=100, stop=1001, step=100)
crossover_probabilities = numpy.arange(start=0, stop=0.91, step=0.1)
mutation_probabilities = numpy.arange(start=0, stop=0.31, step=0.025)
test_repeats = 30

def test():
    graph_file_list = ["./graphs/graph_small.dat", "./graphs/graph_30.dat", "./graphs/graph_tree_30.dat"]
    for graph_file in graph_file_list:
        test_all_parameters_in_single_graph(graph_file)

class TestResult:
    def __init__(
        self,
        id: int,
        graph_name: str,
        cost: float,
        vertex_path: list[int],
        edge_path: list[int],
        population_size: int,
        num_generations: int,
        crossover_probability: float,
        mutation_probability: float,
    ):
        self.__id: int = id
        self.__graph_name: str = graph_name
        self.__cost: float = cost
        self.__vertex_path: list[int] = vertex_path
        self.__edge_path: list[int] = edge_path
        self.__population_size: int = population_size
        self.__num_generations: int = num_generations
        self.__crossover_probability: float = crossover_probability
        self.__mutation_probability: float = mutation_probability

    @property
    def get_cost(self):
        return self.__cost

    def __str__ (self):
        return (
            f"Test {self.__graph_name} {self.__id}:\n"
            f"Vertex path: {self.__vertex_path}\n"
            f"Edge path: {self.__edge_path}\n"
            f"Cost: {self.__cost}\n"
            f"Pop size: {self.__population_size}\n"
            f"Num gen: {self.__num_generations}\n"
            f"Cross prob: {self.__crossover_probability}\n"
            f"Mut prob: {self.__mutation_probability}\n\n"
        )


def test_all_parameters_in_single_graph(graph_file: str):
    test_results: list[TestResult] = []

    graph = read_graph_from_file(graph_file)
    assert(GraphType.of(graph) is GraphType.generic)
    assert(graph.is_connected())

    matrix = create_matrix(graph)
    index = 1

    result_file = open(f"test_results_{time()}.csv", "a")

    for population_size in population_sizes:
        for num_generations in nums_generations:
            for crossover_probability in crossover_probabilities:
                for mutation_probability in mutation_probabilities:
                    for _ in range(test_repeats):
                        print(f"Calculating calc {index} with params: {population_size}, {num_generations} {crossover_probability} {mutation_probability}")

                        ga_instance = create_template_ga_instance(
                            graph=graph,
                            matrix=matrix,
                            population_size=population_size,
                            num_generations=num_generations,
                            crossover_probability=crossover_probability,
                            mutation_probability=mutation_probability,
                        )

                        vertex_path, cost, edge_path = solve(graph, matrix, ga_instance)

                        test_result = TestResult(
                            index,
                            graph_file, 
                            cost,
                            vertex_path,
                            edge_path,
                            population_size,
                            num_generations,
                            crossover_probability,
                            mutation_probability
                        )

                        result_file.write(
                            test_result.__id,
                            test_result.__graph_name,
                            test_result.__cost,
                            test_result.__num_generations,
                            test_result.__population_size,
                            test_result.__crossover_probability,
                            test_result.__mutation_probability
                        )   

                        print(test_result)

                        test_results.append(test_result)
                        index += 1

    result_file.close()
    return test_results


def calculate_indicators(test_results: list[TestResult], test_repeats: int):
    test_result_costs = list(map(lambda test_result: test_result.get_cost, test_results))

    min_cost = min(test_result_costs)
    max_cost = max(test_result_costs)
    avg_cost: float = numpy.mean(test_result_costs)

    # all_tests_results_array = numpy.array(test_result_costs)
    # indicators = numpy.sqrt(
    #     numpy.sum(
    #         all_tests_results_array**2 - min_cost**2, axis=0
    #     ) / test_repeats
    # )
    # best_indicator_index = numpy.argmin(indicators)
    # best_indicator = indicators[best_indicator_index]

    return {
        "min_cost": min_cost,
        "max_cost": max_cost,
        "avg_cost": avg_cost,
        # "best_indicator": best_indicator,
        # "best_result": test_result_costs[best_indicator_index],
    }


def create_csv_file(test_results: list[TestResult]):
    f = open(f"test_results/test_results_{time()}.csv", "a")

    f.write("id, graph_file, cost, num_generations, population_size, crossover_probability, mutation_probability")

    for test_result in test_results:
        f.write(
            test_result.__id,
            test_result.__graph_name,
            test_result.__cost,
            test_result.__num_generations,
            test_result.__population_size,
            test_result.__crossover_probability,
            test_result.__mutation_probability
        )

    f.close()


if __name__ == "__main__":
    test()
