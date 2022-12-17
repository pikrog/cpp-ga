from itertools import groupby

import numpy
import igraph
from pygad import pygad

from cpp.ga.util import pmx_crossover, mutate_by_swap
from cpp.graphutil import generate_random_permutations, PathMatrix


def _fitness(graph: igraph.Graph, matrix: PathMatrix, solution, solution_index):
    total_cost = 0
    for i, edge_id in enumerate(solution):
        edge = graph.es[edge_id]
        i_next = (i + 1) % len(solution)
        next_edge_id = solution[i_next]
        next_edge = graph.es[next_edge_id]
        # total_cost += edge["weight"]
        total_cost += matrix.min_paths_costs[edge.target, next_edge.source]
    return -total_cost


def create_template_ga_instance(
        graph: igraph.Graph,
        matrix: PathMatrix,
        population_size=30, num_generations=100, crossover_probability=0.9, mutation_probability=0.1,
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
    fitness = -solution[1]

    cost = 0
    path = []
    path.append(graph.es[genotype[0]].source)

    for i, edge_id in enumerate(genotype):
        edge = graph.es[edge_id]
        i_next = (i + 1) % len(genotype)
        next_edge_id = genotype[i_next]
        next_edge = graph.es[next_edge_id]
        edge_path = matrix.min_paths[edge.target, next_edge.source]

        if edge not in edge_path:
            path.append(edge.target)
        for e in edge_path:
            e = graph.es[e]
            path += [e.source, e.target]
        if next_edge not in edge_path:
            path.append(next_edge.source)

        # cost += matrix.min_paths_costs[edge.source, next_edge.source]

    path = [v for v, _group in groupby(path)]

    return path, cost, genotype, fitness


def find_path(graph: igraph.Graph, matrix: PathMatrix, ga_instance: pygad.GA):
    ga_instance.run()
    path, cost, phenotype, _ = _interpret_ga_solution(graph, matrix, ga_instance.best_solution())
    return path, cost, phenotype


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
    path, cost, _ = find_path(graph, matrix, ga_instance)
    return path, cost


def calculate_cost(graph: igraph.Graph, vertices: list[int]):
    cost = 0
    for index in range(len(vertices) - 1):
        start_vertex = vertices[index]
        end_vertex = vertices[index + 1]
        cost += graph.es.find(_source=start_vertex, _target=end_vertex)['weight']

    return cost


def create_vertex_path(graph: igraph.Graph, edges: list[int]):
    edge_dataframe = graph.get_edge_dataframe()
    edge_dataframe.loc[len(edge_dataframe.index)] = edge_dataframe.iloc[0]

    vertex_path = []
    for edge in edges:
        source = edge_dataframe.iloc[int(edge)].source
        target = edge_dataframe.iloc[int(edge)].target

        if len(vertex_path) == 0:
            vertex_path.append(source)
            vertex_path.append(target)
        elif edge == edges[-1]:
            previous_vertex = vertex_path[-1]
            path = graph.get_shortest_paths(v=previous_vertex, to=vertex_path[0], weights="weight")[0]
            for vertex in path[1:]:
                vertex_path.append(vertex)
        elif len(vertex_path) > 0:
            previous_vertex = vertex_path[-1]

            if source == previous_vertex:
                vertex_path.append(target)
            elif target == previous_vertex:
                vertex_path.append(source)
            else:
                source_path = graph.get_shortest_paths(v=previous_vertex, to=source, weights="weight")[0]
                target_path = graph.get_shortest_paths(v=previous_vertex, to=target, weights="weight")[0]

                if source in source_path and target in source_path:
                    for vertex in source_path[1:]:
                        vertex_path.append(vertex)
                elif source in target_path and target in target_path:
                    for vertex in target_path[1:]:
                        vertex_path.append(vertex)
                else:
                    source_path_cost = calculate_cost(graph, source_path)
                    target_path_cost = calculate_cost(graph, target_path)
                    better_path = source_path if source_path_cost < target_path_cost else target_path
                    last_vertex = target if source_path_cost < target_path_cost else source

                    for vertex in better_path[1:]:
                        vertex_path.append(vertex)
                    vertex_path.append(last_vertex)
        else:
            vertex_path.append(source)
            vertex_path.append(target)
    return vertex_path
