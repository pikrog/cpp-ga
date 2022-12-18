import numpy
import pygad


def _pmx_insert(parent1, parent2, segment_begin, segment_end):
    child = [None] * len(parent1)

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


def pmx_crossover(parents, offspring_size, ga_instance: pygad.GA):
    offspring = []
    index = 0
    while len(offspring) != offspring_size[0]:
        parent1 = list(parents[index % parents.shape[0], :])
        parent2 = list(parents[(index + 1) % parents.shape[0], :])

        index += 1
        if not try_select(threshold=ga_instance.crossover_probability):
            offspring.append(parent1)
            continue

        segment_begin = numpy.random.choice(range(offspring_size[1]))
        max_segment_length = offspring_size[1] - segment_begin
        segment_end = segment_begin + numpy.random.choice(range(max_segment_length))
        child = _pmx_insert(parent1, parent2, segment_begin, segment_end)
        offspring.append(child)

    return numpy.array(offspring)


def mutate_by_swap(offspring, ga_instance: pygad.GA):
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

# def mutate_by_swap(offspring, ga_instance: pygad.GA):
#     for chromosome_index in range(offspring.shape[0]):
#         for gene_index_1, gene_1 in enumerate(offspring[chromosome_index]):
#             if not try_select(threshold=ga_instance.mutation_probability):
#                 continue
#             gene_index_2 = numpy.random.choice(range(offspring.shape[1]))
#             gene_2 = offspring[chromosome_index, gene_index_2]
#
#             offspring[chromosome_index, gene_index_1] = gene_2
#             offspring[chromosome_index, gene_index_2] = gene_1
#
#     return offspring


def try_select(threshold, low=0.0, high=1.0):
    roulette = numpy.random.uniform(low=low, high=high)
    return roulette <= threshold
