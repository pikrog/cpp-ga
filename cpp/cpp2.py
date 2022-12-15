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
        for vertex in shortest_vertex_path:
            path_cost += line_graph.vs[vertex]["weight"]

        total_cost += path_cost

    path_to_beggining = line_graph.get_shortest_paths(v=int(solution[0]), to=int(solution[-1]))[0]
    going_round_cost = 0
    for vertex in path_to_beggining:
        going_round_cost += line_graph.vs[vertex]["weight"]
    total_cost += going_round_cost


    return -total_cost


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


def create_template_ga_instance(line_graph: igraph.Graph, population_size=50, num_generations=1000,
                                crossover_probability=0.5, mutation_probability=0.5,
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
        keep_elitism=2,
        keep_parents=-1,
        **kwargs)


def solve_cpp(graph: igraph.Graph, ga_instance: pygad.GA | None = None):
    line_graph = graph.linegraph()
    for vertex in range(line_graph.vcount()):
        line_graph.vs[vertex]["weight"] = graph.es[vertex]["weight"]

    if ga_instance is None:
        ga_instance = create_template_ga_instance(line_graph)
    ga_instance.run()
    return ga_instance.best_solution()