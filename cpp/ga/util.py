from enum import Enum

import numpy
import pygad


def try_select(threshold: float, low=0.0, high=1.0):
    roulette = numpy.random.uniform(low=low, high=high)
    return roulette <= threshold


def random_select_elements(threshold: float, size: tuple, low=0.0, high=1.0):
    matrix = numpy.random.uniform(low, high, size)
    return matrix <= threshold


def pmx_insert(parent1: list[int], parent2: list[int], segment_begin: int, segment_end: int):
    child: list[int | None] = [None] * len(parent1)

    parent1_segment = parent1[segment_begin:segment_end]
    parent2_segment = parent2[segment_begin:segment_end]

    child[segment_begin:segment_end] = parent1_segment

    copied_genes = set(parent1_segment)
    for i, gene in enumerate(parent2_segment, start=segment_begin):
        if gene in copied_genes:
            continue
        candidate_in_segment = True
        while candidate_in_segment:
            candidate_gene = parent1[i]
            i = parent2.index(candidate_gene)
            candidate_in_segment = segment_begin <= i < segment_end

        child[i] = gene
        copied_genes.add(gene)

    remaining_genes = (gene for gene in parent2 if gene not in copied_genes)

    for i in range(len(child)):
        if child[i] is None:
            child[i] = next(remaining_genes)

    return child


def abs_pmx_insert(parent1: numpy.array, parent2: numpy.array, segment_begin: int, segment_end: int):
    child: numpy.array = numpy.zeros(shape=len(parent1), dtype=int)

    parent1_segment = parent1[segment_begin:segment_end]
    parent2_segment = parent2[segment_begin:segment_end]

    child[segment_begin:segment_end] = parent1_segment

    copied_genes = set(numpy.abs(parent1_segment))

    abs_parent2 = numpy.abs(parent2)

    parent2_indices = numpy.zeros(shape=len(parent2) + 1, dtype=int)
    for i in range(len(parent2)):
        parent2_indices[abs_parent2[i]] = i

    for i, gene in enumerate(parent2_segment, start=segment_begin):
        abs_gene = abs(gene)
        if abs_gene in copied_genes:
            continue
        candidate_in_segment = True
        while candidate_in_segment:
            candidate_gene = parent1[i]
            i = parent2_indices[abs(candidate_gene)]
            candidate_in_segment = segment_begin <= i < segment_end

        child[i] = gene
        copied_genes.add(abs_gene)

    remaining_genes = [gene for i, gene in enumerate(parent2) if abs_parent2[i] not in copied_genes]

    j = 0
    for i in range(len(child)):
        if child[i] == 0:
            child[i] = remaining_genes[j]
            j += 1

    return child


def abs_pmx_crossover(parents: numpy.array, offspring_size: tuple[int, int], ga_instance: pygad.GA):
    return pmx_crossover(parents, offspring_size, ga_instance, abs_pmx_insert)


def pmx_crossover(parents: numpy.array, offspring_size: tuple[int, int], ga_instance: pygad.GA, inserter=pmx_insert):
    offspring = numpy.ndarray(shape=offspring_size, dtype=int)
    offspring_index = 0
    parent_index = 0

    while offspring_index != offspring_size[0]:
        parent1 = list(parents[parent_index % parents.shape[0], :])
        parent_index += 1

        if not try_select(threshold=ga_instance.crossover_probability):
            offspring[offspring_index][:] = parent1
            offspring_index += 1
            continue

        parent2_index = numpy.random.randint(0, parents.shape[0], size=1)[0]
        # parent2_index = parent_index % parents.shape[0]
        parent2 = list(parents[parent2_index, :])

        segment_begin = numpy.random.choice(range(offspring_size[1]))
        max_segment_length = offspring_size[1] - segment_begin
        segment_end = segment_begin + numpy.random.choice(range(max_segment_length))

        child = inserter(parent1, parent2, segment_begin, segment_end)

        offspring[offspring_index][:] = child
        offspring_index += 1

    return offspring


def mutate_by_negation_or_swap(offspring: numpy.array, ga_instance: pygad.GA):
    chances = numpy.random.uniform(size=offspring.shape)
    swapped_genes = numpy.random.uniform(size=offspring.shape) <= 0.5
    genes_to_mutate = chances <= ga_instance.mutation_probability
    genes_to_swap = numpy.full(shape=offspring.shape, fill_value=False, dtype=bool)
    genes_to_swap[genes_to_mutate] = swapped_genes[genes_to_mutate]
    genes_to_negate = numpy.full(shape=offspring.shape, fill_value=False, dtype=bool)
    genes_to_negate[genes_to_mutate] = ~swapped_genes[genes_to_mutate]
    offspring[genes_to_negate] *= -1
    num_genes_to_swap = numpy.count_nonzero(genes_to_swap)
    base_swap_indices = numpy.argwhere(genes_to_swap == True)
    random_swap_indices = numpy.random.choice(range(offspring.shape[1]), size=num_genes_to_swap)
    for i, (x, y) in enumerate(base_swap_indices):
        offspring[x, random_swap_indices[i]], offspring[x, y] = offspring[x, y], offspring[x, random_swap_indices[i]]
    return offspring


def mutate_by_negation(offspring: numpy.array, ga_instance: pygad.GA):
    selected_to_negate = random_select_elements(threshold=ga_instance.mutation_probability, size=offspring.shape)
    offspring[selected_to_negate] *= -1
    return offspring


def mutate_by_swap_per_chromosome(offspring: numpy.array, ga_instance: pygad.GA):
    for chromosome_index in range(offspring.shape[0]):
        if not try_select(threshold=ga_instance.mutation_probability):
            continue

        gene_index_1 = numpy.random.choice(range(offspring.shape[1]))
        gene_index_2 = numpy.random.choice(range(offspring.shape[1]))

        gene_value_1 = offspring[chromosome_index, gene_index_1]
        gene_value_2 = offspring[chromosome_index, gene_index_2]

        offspring[chromosome_index, gene_index_1] = gene_value_2
        offspring[chromosome_index, gene_index_2] = gene_value_1

    return offspring
