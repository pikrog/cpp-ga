import pygad
import numpy
import random

from cpp.matrix import OddVerticesPathMatrix


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
    for pair_index in range(int(len(solution) / 2)):
        pair_begin, pair_end = _evaluate_element_indices(pair_index)
        vertex_begin = int(solution[pair_begin])
        vertex_end = int(solution[pair_end])
        path = matrix.min_paths[vertex_begin, vertex_end]
        total_cost += path[0]

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

        if not _try_select(threshold=ga_instance.crossover_probability):
            offspring.append(parent1)
            index += 1
            continue

        pair_index = numpy.random.choice(range(int(offspring_size[1] / 2)))
        child = _insert_pair(parent1, parent2, pair_index)
        offspring.append(child)

        index += 1

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


def _interpret_ga_solution(matrix, solution):
    genotype = [int(gene) for gene in solution[0]]
    phenotype = [matrix.odd_vertices[v] for v in genotype]
    cost = -solution[1]

    paths = []
    for pair_index in range(int(len(genotype) / 2)):
        pair_begin, pair_end = _evaluate_element_indices(pair_index)
        path = matrix.min_paths[genotype[pair_begin], genotype[pair_end]]
        paths.append(path[1])

    return paths, cost, phenotype


def generate_random_permutations(len_permutation, num_permutations):
    population = [list(range(len_permutation)) for _ in range(num_permutations)]
    for solution in population:
        random.shuffle(solution)
    return population


def create_template_ga_instance(
        matrix: OddVerticesPathMatrix,
        population_size=30, num_generations=100, crossover_probability=0.9, mutation_probability=0.1,
        **kwargs
):
    num_genes = len(matrix.odd_vertices)
    initial_population = generate_random_permutations(num_genes, population_size)
    return pygad.GA(
        num_generations=num_generations,
        num_parents_mating=population_size // 2,
        crossover_probability=crossover_probability,
        crossover_type=_crossover,
        mutation_probability=mutation_probability,
        mutation_percent_genes="default",
        mutation_type=_mutate,
        fitness_func=lambda sol, index: _fitness(matrix, sol, index),
        parent_selection_type="rws",  # roulette
        initial_population=initial_population,
        keep_elitism=2,
        keep_parents=-1,
        **kwargs
    )


def find_euler_transform_by_ga(matrix: OddVerticesPathMatrix, ga_instance: pygad.GA):
    ga_instance.run()
    paths, cost, phenotype = _interpret_ga_solution(matrix, ga_instance.best_solution())
    return paths, cost, phenotype
