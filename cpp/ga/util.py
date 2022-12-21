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


def abs_pmx_insert(parent1: list[int], parent2: list[int], segment_begin: int, segment_end: int):
    child: list[int | None] = [None] * len(parent1)

    parent1_segment = parent1[segment_begin:segment_end]
    parent2_segment = parent2[segment_begin:segment_end]

    child[segment_begin:segment_end] = parent1_segment

    # copied_genes = set(map(abs, parent1_segment))
    copied_genes = [False] * (len(parent2) + 1)
    for gene in parent1_segment:
        copied_genes[abs(gene)] = True

    # abs_parent1 = list(map(abs, parent1))
    abs_parent2 = list(map(abs, parent2))

    parent2_indices = [0] * (len(parent2) + 1)
    for i in range(len(parent2)):
        parent2_indices[abs_parent2[i]] = i

    for i, gene in enumerate(parent2_segment, start=segment_begin):
        abs_gene = abs(gene)
        if copied_genes[abs_gene] is True:
            continue
        candidate_in_segment = True
        while candidate_in_segment:
            candidate_gene = parent1[i]
            i = parent2_indices[abs(candidate_gene)]
            candidate_in_segment = segment_begin <= i < segment_end

        child[i] = gene
        copied_genes[abs_gene] = True

    remaining_genes = (gene for gene in parent2 if copied_genes[abs(gene)] is False)

    for i in range(len(child)):
        if child[i] is None:
            child[i] = next(remaining_genes)

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
    for chromosome_index in range(offspring.shape[0]):
        for gene_index_1, gene_1 in enumerate(offspring[chromosome_index]):
            if not try_select(threshold=ga_instance.mutation_probability):
                continue

            if try_select(threshold=0.5):
                gene_index_2 = numpy.random.choice(range(offspring.shape[1]))
                gene_2 = offspring[chromosome_index, gene_index_2]

                offspring[chromosome_index, gene_index_1] = gene_2
                offspring[chromosome_index, gene_index_2] = gene_1
            else:
                offspring[chromosome_index, gene_index_1] *= -1

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
