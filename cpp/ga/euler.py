import igraph
import numpy
import pygad

from cpp.ga.util import mutate_by_swap, try_select
from cpp.graphutil import GraphType, fix_half_euler_graph, duplicate_edges_on_paths, fleury, \
    generate_random_permutations, PathMatrix


# class OddVerticesPathMatrix:
#     def __init__(self, graph: igraph.Graph):
#         # find odd vertices
#         vertices = list(range(graph.vcount()))
#         self.__odd_vertices = list(filter(lambda v: graph.degree(v) % 2 != 0, vertices))
#         num_odd = len(self.__odd_vertices)
#
#         # initialize matrix
#         self.__min_paths = numpy.ndarray(shape=(num_odd, num_odd), dtype=tuple)
#
#         # evaluate path cost matrix
#         for i, j in numpy.ndindex(self.__min_paths.shape):
#             paths = graph.get_shortest_paths(
#                 self.__odd_vertices[i],
#                 to=self.__odd_vertices[j],
#                 weights=graph.es["weight"],
#                 output="epath"
#             )
#             path = paths[0]
#             distance = 0
#             for edge in path:
#                 distance += graph.es[edge]["weight"]
#
#             self.__min_paths[i, j] = (distance, path)
#
#     @property
#     def min_paths(self):
#         return self.__min_paths
#
#     @property
#     def odd_vertices(self):
#         return self.__odd_vertices


def _evaluate_element_indices(pair_index):
    pair_begin = pair_index * 2
    pair_end = pair_begin + 1
    return pair_begin, pair_end


def _insert_pair(source, target, pair_index):
    pair_begin, pair_end = _evaluate_element_indices(pair_index)

    pair = source[pair_begin:pair_end + 1]

    first_swap_index = target.index(pair[0])
    target[first_swap_index] = target[pair_begin]
    target[pair_begin] = pair[0]

    second_swap_index = target.index(pair[1])
    target[second_swap_index] = target[pair_end]
    target[pair_end] = pair[1]

    return target


def _fitness(matrix, solution, solution_index):
    total_cost = 0
    for pair_index in range(len(solution) // 2):
        pair_begin, pair_end = _evaluate_element_indices(pair_index)
        vertex_begin = int(solution[pair_begin])
        vertex_end = int(solution[pair_end])
        cost = matrix.min_paths_costs[vertex_begin, vertex_end]
        total_cost += cost

    return -total_cost


def _crossover(parents, offspring_size, ga_instance: pygad.GA):
    offspring = []
    index = 0
    while len(offspring) != offspring_size[0]:
        parent1 = list(parents[index % parents.shape[0], :])
        parent2 = list(parents[(index + 1) % parents.shape[0], :])

        if not try_select(threshold=ga_instance.crossover_probability):
            offspring.append(parent1)
            index += 1
            continue

        pair_index = numpy.random.choice(range(int(offspring_size[1] / 2)))
        child = _insert_pair(parent1, parent2, pair_index)
        offspring.append(child)

        index += 1

    return numpy.array(offspring)


def _interpret_ga_solution(matrix: PathMatrix, solution):
    genotype = [gene for gene in solution[0]]
    phenotype = [matrix.vertices[v] for v in genotype]
    cost = -solution[1]

    paths = []
    for pair_index in range(len(genotype) // 2):
        pair_begin, pair_end = _evaluate_element_indices(pair_index)
        path = matrix.min_paths[genotype[pair_begin], genotype[pair_end]]
        paths.append(path)

    return paths, cost, phenotype


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
        crossover_type=_crossover,
        mutation_probability=mutation_probability,
        mutation_percent_genes="default",
        mutation_type=mutate_by_swap,
        fitness_func=lambda sol, index: _fitness(matrix, sol, index),
        parent_selection_type="rws",  # roulette
        initial_population=initial_population,
        keep_elitism=2,
        keep_parents=-1,
        gene_type=int,
        **kwargs
    )


def find_euler_transform(matrix: PathMatrix, ga_instance: pygad.GA):
    ga_instance.run()
    paths, cost, phenotype = _interpret_ga_solution(matrix, ga_instance.best_solution())
    return paths, cost, phenotype


def create_matrix(graph: igraph.Graph):
    vertices = list(range(graph.vcount()))
    odd_vertices = list(filter(lambda v: graph.degree(v) % 2 != 0, vertices))
    return PathMatrix(graph, odd_vertices)


def solve(
    graph: igraph.Graph,
    matrix: PathMatrix | None = None,
    ga_instance: pygad.GA | None = None
):
    cost = 0
    graph_type = GraphType.of(graph)
    if graph_type is GraphType.euler:
        euler_graph = graph
    elif graph_type is GraphType.half_euler:
        euler_graph = fix_half_euler_graph(graph)
    else:
        if matrix is None:
            matrix = create_matrix(graph)
        if ga_instance is None:
            ga_instance = create_template_ga_instance(matrix)
        paths, cost, _ = find_euler_transform(matrix, ga_instance)
        euler_graph, _ = duplicate_edges_on_paths(graph, paths)
    path_vertices = fleury(euler_graph)
    return path_vertices, euler_graph, graph_type, cost
