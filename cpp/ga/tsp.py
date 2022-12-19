from itertools import groupby

import numpy
import igraph
from pygad import pygad

from cpp.ga.util import pmx_crossover, mutate_by_swap, random_select_elements, abs_pmx_crossover
from cpp.graphutil import generate_random_permutations, PathMatrix


def _get_vertexes(edge, edge_id):
    return (edge.source, edge.target) if edge_id > 0 else (edge.target, edge.source)


def _fitness(graph: igraph.Graph, matrix: PathMatrix, solution, solution_index):
    total_cost = 0
    first_edge_id = solution[0]
    begin_vertex = graph.es[abs(first_edge_id) - 1].source
    current_vertex = begin_vertex
    for i, edge_id in enumerate(solution):
        edge = graph.es[abs(edge_id) - 1]
        target_vertex, next_vertex = _get_vertexes(edge, edge_id)
        total_cost += matrix.min_paths_costs[current_vertex, target_vertex]
        total_cost += edge["weight"]
        current_vertex = next_vertex
    total_cost += matrix.min_paths_costs[current_vertex, begin_vertex]
    return 1/total_cost


def _generate_random_solutions(num_genes, population_size):
    solutions = numpy.array(generate_random_permutations(num_genes, population_size))
    solutions += 1
    selected_to_negate = random_select_elements(threshold=0.5, size=(population_size, num_genes))
    solutions[selected_to_negate] *= -1
    return solutions


def create_template_ga_instance(
        graph: igraph.Graph,
        matrix: PathMatrix,
        population_size=30, num_generations=1500, crossover_probability=0.9, mutation_probability=0.022,
        num_parents_mating=25,
        **kwargs
):
    num_genes = graph.ecount()
    initial_population = generate_random_permutations(num_genes, population_size)
    return pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        crossover_probability=crossover_probability,
        crossover_type=abs_pmx_crossover,
        mutation_probability=mutation_probability,
        mutation_percent_genes="default",
        mutation_type=mutate_by_swap,
        fitness_func=lambda sol, index: _fitness(graph, matrix, sol, index),
        parent_selection_type="rws",  # roulette
        initial_population=initial_population,
        keep_elitism=2,
        keep_parents=-1,
        gene_type=int,
        **kwargs
    )


def _interpret_ga_solution(graph: igraph.Graph, matrix: PathMatrix, solution):
    genotype = [gene for gene in solution[0]]

    cost = 0
    edge_path = []

    first_edge_id = genotype[0]
    first_edge = graph.es[abs(first_edge_id) - 1]
    current_vertex = begin_vertex = first_edge.source
    for i, edge_id in enumerate(genotype):
        edge = graph.es[abs(edge_id) - 1]
        target_vertex, next_vertex = _get_vertexes(edge, edge_id)
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

    prev_vertex = begin_vertex
    vertex_path = [prev_vertex]
    for edge_index in edge_path:
        edge = graph.es[edge_index]
        vertex1, vertex2 = edge.source, edge.target
        if prev_vertex == vertex1:
            next_vertex = vertex2
        else:
            next_vertex = vertex1
        vertex_path.append(next_vertex)
        prev_vertex = next_vertex
    edge_path = list(map(lambda e: (graph.es[e].source, graph.es[e].target), edge_path))
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

