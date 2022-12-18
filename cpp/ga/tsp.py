from itertools import groupby

import numpy
import igraph
from pygad import pygad

from cpp.ga.util import pmx_crossover, mutate_by_swap
from cpp.graphutil import generate_random_permutations, PathMatrix


def _fitness(graph: igraph.Graph, matrix: PathMatrix, solution, solution_index):
    total_cost = 0
    first_edge_id = solution[0]
    begin_vertex = graph.es[first_edge_id].source
    current_vertex = begin_vertex
    for i, edge_id in enumerate(solution):
        edge = graph.es[edge_id]
        from_vertex_to_source = matrix.min_paths_costs[current_vertex, edge.source]
        from_vertex_to_target = matrix.min_paths_costs[current_vertex, edge.target]
        if from_vertex_to_source < from_vertex_to_target:
            current_vertex = edge.target
            total_cost += from_vertex_to_source
        else:
            current_vertex = edge.source
            total_cost += from_vertex_to_target
        total_cost += edge["weight"]
    total_cost += matrix.min_paths_costs[current_vertex, begin_vertex]
    return -total_cost


def create_template_ga_instance(
        graph: igraph.Graph,
        matrix: PathMatrix,
        population_size=50, num_generations=300, crossover_probability=0.98, mutation_probability=0.5,
        **kwargs
):
    num_genes = graph.ecount()
    initial_population = generate_random_permutations(num_genes, population_size)
    return pygad.GA(
        num_generations=num_generations,
        num_parents_mating=population_size // 2,
        crossover_probability=crossover_probability,
        crossover_type=pmx_crossover,
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
    # phenotype = [graph.edge for v in genotype]

    cost = -solution[1]
    edge_path = []

    first_edge_id = genotype[0]
    first_edge = graph.es[first_edge_id]
    current_vertex = begin_vertex = first_edge.source
    for i, edge_id in enumerate(genotype):
        edge = graph.es[edge_id]
        from_vertex_to_source = matrix.min_paths_costs[current_vertex, edge.source]
        from_vertex_to_target = matrix.min_paths_costs[current_vertex, edge.target]

        if from_vertex_to_source < from_vertex_to_target:
            sub_path = matrix.min_paths[current_vertex, edge.source]
            current_vertex = edge.target
        else:
            sub_path = matrix.min_paths[current_vertex, edge.target]
            current_vertex = edge.source

        edge_path += sub_path
        edge_path.append(edge.index)

    final_path = matrix.min_paths[current_vertex, begin_vertex]
    edge_path += final_path

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
    return vertex_path, cost, genotype, edge_path


def find_path(graph: igraph.Graph, matrix: PathMatrix, ga_instance: pygad.GA):
    ga_instance.run()
    vertex_path, cost, phenotype, edge_path = _interpret_ga_solution(graph, matrix, ga_instance.best_solution())
    return vertex_path, cost, phenotype, edge_path


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
    vertex_path, cost, phenotype, edge_path = find_path(graph, matrix, ga_instance)
    return vertex_path, cost, phenotype, edge_path

