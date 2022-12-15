from __future__ import annotations

import random

import igraph
import numpy
import pygad


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


def _fitness(line_graph: igraph.Graph, solution, solution_index):
    total_cost = 0
    for index in range(len(solution)):
        begin_vertex = int(solution[index])
        end_vertex = int(solution[(index + 1) % len(solution)])
        # print(begin_vertex, end_vertex, solution)
        shortest_vertex_path = line_graph.get_shortest_paths(v=begin_vertex, to=end_vertex)[0]

        path_cost = 0
        for vertex in shortest_vertex_path[1:]:
            path_cost += line_graph.vs[vertex]["weight"]

        total_cost += path_cost

    return -numpy.log(total_cost)


def _fitness2(graph: igraph.Graph, solution: list[float], solution_index):
    vertex_path = create_vertex_path(graph, solution)

    return -calculate_cost(graph, vertex_path)


def _try_select(threshold, low=0.0, high=1.0):
    roulette = numpy.random.uniform(low=low, high=high)
    return roulette <= threshold


def _crossover(parents, offspring_size, ga_instance: pygad.GA):
    offspring = []
    index = 0
    while len(offspring) != offspring_size[0]:
        parent1 = list(parents[index % parents.shape[0], :])
        parent2 = list(parents[(index + 1) % parents.shape[0], :])

        index += 1
        if not _try_select(threshold=ga_instance.crossover_probability):
            offspring.append(parent1)
            continue

        pair_index = numpy.random.choice(range(int(offspring_size[1] / 2)))
        child = _insert_pair(parent1, parent2, pair_index)
        offspring.append(child)

    return numpy.array(offspring)


def _mutate(offspring, ga_instance: pygad.GA):
    for chromosome_index in range(offspring.shape[0]):
        if not _try_select(threshold=ga_instance.mutation_probability):
            continue

        gene_index_1 = numpy.random.choice(range(offspring.shape[1]))
        gene_index_2 = numpy.random.choice(range(offspring.shape[1]))

        gene_value_1 = offspring[chromosome_index, gene_index_1]
        gene_value_2 = offspring[chromosome_index, gene_index_2]

        offspring[chromosome_index, gene_index_1] = gene_value_2
        offspring[chromosome_index, gene_index_2] = gene_value_1

    return offspring


def generate_initial_population(edge_count: int, population_size: int):
    population = [list(range(edge_count)) for _ in range(population_size)]
    for solution in population:
        random.shuffle(solution)
    return population


def create_template_ga_instance(line_graph: igraph.Graph, population_size=100, num_generations=1000,
                                crossover_probability=0.5, mutation_probability=0.1,
                                **kwargs):
    num_genes = line_graph.vcount()
    initial_population = generate_initial_population(num_genes, population_size)
    return pygad.GA(
        num_generations=num_generations,
        num_parents_mating=population_size // 2,
        crossover_probability=crossover_probability,
        crossover_type=_crossover,
        mutation_probability=mutation_probability,
        mutation_percent_genes="default",
        mutation_type=_mutate,
        fitness_func=lambda solution, index: _fitness(line_graph, solution, index),
        parent_selection_type="rws",  # roulette
        initial_population=initial_population,
        keep_elitism=10,
        keep_parents=-1,
        **kwargs)


def solve_cpp(graph: igraph.Graph, ga_instance: pygad.GA | None = None):
    line_graph = graph.linegraph()
    for vertex in range(line_graph.vcount()):
        line_graph.vs[vertex]["weight"] = graph.es[vertex]["weight"]

    if ga_instance is None:
        ga_instance = create_template_ga_instance(line_graph)
    ga_instance.run()

    edges, score, index = ga_instance.best_solution()
    edges: list[float] = list(edges)
    score: int = -_fitness2(graph, edges, 0)
    # print(edges)
    vertices_path = create_vertex_path(graph, edges)

    return score, vertices_path


def calculate_cost(graph: igraph.Graph, vertices: list[int]):
    cost = 0
    for index in range(len(vertices) - 1):
        start_vertex = vertices[index]
        end_vertex = vertices[index + 1]
        cost += graph.es.find(_source=start_vertex, _target=end_vertex)['weight']

    return cost


def create_vertex_path(graph: igraph.Graph, edges: list[float]):
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
