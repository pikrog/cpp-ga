import igraph
import numpy
from pygad import pygad

from cpp.ga.util import random_select_elements, abs_pmx_crossover, mutate_by_negation_or_swap
from cpp.graphutil import generate_random_permutations, PathMatrix, edge_to_vertex_path


def _decode_gene(graph: igraph.Graph, gene: int):
    edge = graph.es[abs(gene) - 1]
    source, target = (edge.source, edge.target) if gene > 0 else (edge.target, edge.source)
    return edge, source, target


def _fitness(graph: igraph.Graph, matrix: PathMatrix, solution: list[int]):
    total_cost = 0
    first_edge_id = solution[0]
    begin_vertex = graph.es[abs(first_edge_id) - 1].source
    current_vertex = begin_vertex
    for gene in solution:
        edge = graph.es[abs(gene) - 1]
        target_vertex, next_vertex = (edge.source, edge.target) if gene > 0 else (edge.target, edge.source)
        total_cost += matrix.min_paths_costs[current_vertex, target_vertex]
        total_cost += edge["weight"]
        current_vertex = next_vertex
    total_cost += matrix.min_paths_costs[current_vertex, begin_vertex]
    return 1/total_cost


def _generate_random_solutions(num_genes: int, population_size: int):
    solutions = numpy.array(generate_random_permutations(num_genes, population_size))
    solutions += 1
    selected_to_negate = random_select_elements(threshold=0.5, size=(population_size, num_genes))
    solutions[selected_to_negate] *= -1
    return solutions


def create_template_ga_instance(
        graph: igraph.Graph,
        matrix: PathMatrix,
        population_size=50, num_generations=300, crossover_probability=0.9, mutation_probability=0.022,
        num_parents_mating=40,
        keep_elitism=2, parent_selection_type="rws",    # roulette
        **kwargs
):
    num_genes = graph.ecount()
    initial_population = _generate_random_solutions(num_genes, population_size)
    return pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        crossover_probability=crossover_probability,
        crossover_type=abs_pmx_crossover,
        mutation_probability=mutation_probability,
        mutation_type=mutate_by_negation_or_swap,
        fitness_func=lambda sol, index: _fitness(graph, matrix, sol),
        parent_selection_type=parent_selection_type,
        initial_population=initial_population,
        keep_elitism=keep_elitism,
        gene_type=int,
        **kwargs
    )


def _interpret_ga_solution(graph: igraph.Graph, matrix: PathMatrix, solution: tuple[list[int], int]):
    genotype = [gene for gene in solution[0]]

    cost = 0
    edge_path = []

    first_gene = genotype[0]
    first_edge, _, _ = _decode_gene(graph, first_gene)
    current_vertex = begin_vertex = first_edge.source
    for gene in genotype:
        edge, target_vertex, next_vertex = _decode_gene(graph, gene)
        sub_path = matrix.min_paths[current_vertex, target_vertex]
        sub_path_cost = matrix.min_paths_costs[current_vertex, target_vertex]

        edge_path += sub_path
        cost += sub_path_cost
        edge_path.append(edge.index)
        cost += edge["weight"]

        current_vertex = next_vertex

    final_path = matrix.min_paths[current_vertex, begin_vertex]
    final_path_cost = matrix.min_paths_costs[current_vertex, begin_vertex]
    edge_path += final_path
    cost += final_path_cost

    edge_path = list(map(lambda e: (graph.es[e].source, graph.es[e].target), edge_path))
    vertex_path = edge_to_vertex_path(edge_path, begin_vertex)

    return vertex_path, cost, edge_path


def find_path(graph: igraph.Graph, matrix: PathMatrix, ga_instance: pygad.GA):
    ga_instance.run()
    vertex_path, cost, edge_path = _interpret_ga_solution(graph, matrix, ga_instance.best_solution())
    return vertex_path, cost, edge_path


def create_matrix(graph: igraph.Graph):
    return PathMatrix(graph)


def solve(
        graph: igraph.Graph,
        matrix: PathMatrix | None = None,
        ga_instance: pygad.GA | None = None
):
    if matrix is None:
        matrix = create_matrix(graph)
    if ga_instance is None:
        ga_instance = create_template_ga_instance(graph, matrix)
    vertex_path, cost, edge_path = find_path(graph, matrix, ga_instance)
    return vertex_path, cost, edge_path

