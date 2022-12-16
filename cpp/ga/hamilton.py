import numpy
import igraph
from pygad import pygad

from cpp.ga.util import pmx_crossover, mutate_by_swap
from cpp.graphutil import generate_random_permutations


class PathMatrix:
    def __init__(self, graph: igraph.Graph):
        self.__vertices = list(range(graph.vcount()))
        num_vertices = len(self.__vertices)

        self.__min_paths = numpy.ndarray(shape=(num_vertices, num_vertices))

        for v in self.__vertices:
            paths = graph.get_shortest_paths(
                v,
                to=range(len(v)),
                weights=graph.es["weight"],
                output="epath"
            )

            distances = numpy.ndarray(shape=(1, num_vertices))
            for i, path in enumerate(paths):
                distance = 0
                for edge in path:
                    distance += graph.es[edge]["weight"]
                distances[i] = distance

            self.__min_paths[v][:] = distances

        self.__min_paths += numpy.transpose(self.__min_paths)

    @property
    def min_paths(self):
        return self.__min_paths

    @property
    def vertices(self):
        return self.__vertices


def _fitness(matrix: PathMatrix, solution, solution_index):

    pass


def create_template_ga_instance(
        matrix: PathMatrix,
        population_size=30, num_generations=100, crossover_probability=0.9, mutation_probability=0.1,
        **kwargs
):
    num_genes = len(matrix.vertices)
    initial_population = generate_random_permutations(num_genes, population_size)
    return pygad.GA(
        num_generations=num_generations,
        num_parents_mating=population_size // 2,
        crossover_probability=crossover_probability,
        crossover_type=pmx_crossover,
        mutation_probability=mutation_probability,
        mutation_percent_genes="default",
        mutation_type=mutate_by_swap,
        fitness_func=lambda sol, index: _fitness(matrix, sol, index),
        parent_selection_type="rws",  # roulette
        initial_population=initial_population,
        keep_elitism=2,
        keep_parents=-1,
        gene_type=float,
        **kwargs
    )


def _interpret_ga_solution(matrix: PathMatrix, solution):
    pass


def find_path(matrix: PathMatrix, ga_instance: pygad.GA):
    ga_instance.run()
    path, cost, phenotype = _interpret_ga_solution(matrix, ga_instance.best_solution())
    return path, cost, phenotype


def solve(
        graph: igraph.Graph,
        matrix: PathMatrix | None = None,
        ga_instance: pygad.GA | None = None
):
    if matrix is None:
        matrix = PathMatrix(graph)
    if ga_instance is None:
        ga_instance = create_template_ga_instance(matrix)
    path_vertices = []

    return path_vertices, cost
