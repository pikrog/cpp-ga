import pygad
import numpy
import igraph
import random
import matplotlib.pyplot as pyplot


def evaluate_element_indices(pair_index):
    pair_begin = pair_index * 2
    pair_end = pair_begin + 1
    return pair_begin, pair_end


class OddVerticesPathMatrix:
    def __init__(self, graph: igraph.Graph):
        # find odd vertices
        vertices = list(range(graph.vcount()))
        self.__odd_vertices = list(filter(lambda v: graph.degree(v) % 2 != 0, vertices))
        num_odd = len(self.__odd_vertices)

        # initialize matrix
        self.__min_paths = numpy.ndarray(shape=(num_odd, num_odd), dtype=tuple)

        # evaluate path cost matrix
        for i, j in numpy.ndindex(self.__min_paths.shape):
            paths = graph.get_shortest_paths(
                self.__odd_vertices[i],
                to=self.__odd_vertices[j],
                weights=graph.es["weight"],
                output="epath"
            )
            path = paths[0]
            distance = 0
            for edge in path:
                distance += graph.es[edge]["weight"]

            self.__min_paths[i, j] = (distance, path)

    @property
    def min_paths(self):
        return self.__min_paths

    @property
    def odd_vertices(self):
        return self.__odd_vertices


def insert_pair(source, target, pair_index):
    pair_begin, pair_end = evaluate_element_indices(pair_index)

    pair = source[pair_begin:pair_end + 1]

    first_swap_index = target.index(pair[0])
    target[first_swap_index] = target[pair_begin]
    target[pair_begin] = pair[0]

    second_swap_index = target.index(pair[1])
    target[second_swap_index] = target[pair_end]
    target[pair_end] = pair[1]

    return target


def fitness(matrix, solution, solution_index):
    total_cost = 0
    for pair_index in range(int(len(solution) / 2)):
        pair_begin, pair_end = evaluate_element_indices(pair_index)
        vertex_begin = int(solution[pair_begin])
        vertex_end = int(solution[pair_end])
        path = matrix.min_paths[vertex_begin, vertex_end]
        total_cost += path[0]

    return -total_cost


def try_select(threshold, low=0.0, high=1.0):
    roulette = numpy.random.uniform(low=low, high=high)
    return roulette <= threshold


# todo: crossover probability
# https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/master/pygad.py
def crossover(parents, offspring_size, ga_instance: pygad.GA):
    offspring = []
    index = 0
    while len(offspring) != offspring_size[0]:
        parent1 = list(parents[index % parents.shape[0], :])
        parent2 = list(parents[(index + 1) % parents.shape[0], :])

        if not try_select(threshold=ga_instance.crossover_probability):
            offspring.append(parent1)
            index += 1
            continue

        pair_index = numpy.random.choice(range(int(offspring_size[1] / 2)))
        child = insert_pair(parent1, parent2, pair_index)
        offspring.append(child)

        index += 1

    return numpy.array(offspring)


def mutate(offspring, ga_instance: pygad.GA):
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


def make_graph(weighted_edges):
    edges = [edge[0:2] for edge in weighted_edges]
    weights = [edge[2] for edge in weighted_edges]

    return igraph.Graph(
        edges=edges,
        edge_attrs={'weight': weights}
    )


def read_graph_from_file(path):
    with open(path) as file:
        lines = file.readlines()
        weighted_edges = []
        for line in lines:
            strings = line.rstrip().split(' ')
            values = [int(s) for s in strings]
            weighted_edges.append(values)

        return make_graph(weighted_edges)


def generate_random_permutations(len_permutation, num_permutations):
    population = [list(range(len_permutation)) for _ in range(num_permutations)]
    for solution in population:
        random.shuffle(solution)
    return population


def interpret_ga_solution(matrix, solution):
    genotype = [int(gene) for gene in solution[0]]
    phenotype = [matrix.odd_vertices[v] for v in genotype]
    cost = solution[1]

    paths = []
    for pair_index in range(int(len(genotype) / 2)):
        pair_begin, pair_end = evaluate_element_indices(pair_index)
        path = matrix.min_paths[genotype[pair_begin], genotype[pair_end]]
        paths.append(path[1])

    return paths, cost, phenotype


def duplicate_edges_on_paths(graph, paths):
    new_edges = []
    new_graph = graph.copy()
    for edge_indices in paths:
        edges = []
        for edge_index in edge_indices:
            edge = graph.es[edge_index]
            new_graph.add_edge(edge.source, edge.target, weight=edge["weight"])
            edges.append((edge.source, edge.target))
        new_edges.append(edges)
    return new_graph, new_edges


def is_bridge(graph: igraph.Graph, edge: tuple[int, int]):
    chosen_edge = graph.es.find(_source=edge[0], _target=edge[1])
    graph.delete_edges(chosen_edge)
    res = graph.get_all_simple_paths(edge[0], edge[1])
    graph.add_edges([(edge[0], edge[1])])

    if len(res) == 0:
        return True
    else:
        return False


def choose_edge(graph: igraph.Graph, visited_edges: list[tuple[int, int]], vertex: int) -> tuple[int, int]:
    copy: igraph.Graph = graph.copy()

    for i in range(len(visited_edges)):
        copy.delete_edges(copy.es.find(_source=visited_edges[i][0], _target=visited_edges[i][1]))

    all_possible_edges = list(map(lambda x: (vertex, x), graph.get_adjlist()[vertex]))
    available_non_visited_edges = list(map(lambda x: (vertex, x), copy.get_adjlist()[vertex]))
    available_non_bridges = list(filter(lambda x: (not is_bridge(copy, x)), available_non_visited_edges))

    if len(available_non_bridges) > 0:
        return available_non_bridges[0]
    elif len(available_non_visited_edges) > 0:
        return available_non_visited_edges[0]
    else:
        return all_possible_edges[0]


def check_if_valid(graph: igraph.Graph):
    assert (graph.is_connected())
    for vertex in graph.get_adjacency():
        assert (len(vertex) % 2 == 0)


def get_odd_vertices(graph: igraph.Graph):
    vertices = list(range(graph.vcount()))
    return list(filter(lambda v: graph.degree(v) % 2 != 0, vertices))


def is_euler_graph(graph: igraph.Graph):
    odd_vertices = list(filter(lambda v: graph.degree(v) % 2 != 0, list(range(graph.vcount()))))
    return graph.is_connected() and len(odd_vertices) == 0


def get_eulerian_path(graph: igraph.Graph):
    vcount = graph.vcount()

    for i in range(vcount):
        for j in range(vcount):
            if i == j:
                continue

            all_simple_paths = graph.get_all_simple_paths(i, j)
            eulerian_path = list(filter(lambda x: len(x) == vcount, all_simple_paths))

            if len(eulerian_path) > 0:
                return eulerian_path[0]
    return []


def is_half_euler_graph(graph: igraph.Graph):
    odd_vertices = get_odd_vertices(graph)

    if len(odd_vertices) > 2:
        return False

    eulerian_path = get_eulerian_path(graph)

    if len(eulerian_path) > 0:
        return True
    return False


def fleury(graph: igraph.Graph):
    check_if_valid(graph)
    visited_edges: list[tuple[int, int]] = []
    current_vertex = list(graph.get_vertex_dataframe().index)[0]
    visited_vertices: list[int] = [current_vertex]

    while len(visited_edges) < graph.ecount():
        chosen_edge = choose_edge(graph, visited_edges, current_vertex)
        visited_vertices.append(chosen_edge[1])
        current_vertex = chosen_edge[1]
        visited_edges.append(chosen_edge)

    return visited_vertices


def fix_half_euler_graph(graph: igraph.Graph) -> igraph.Graph:
    eulerian_path = get_eulerian_path(graph)
    shortest_path = graph.get_shortest_paths(eulerian_path[0], eulerian_path[-1])[0]
    edges_to_add: list[tuple[int, int]] = []

    for i in range(len(shortest_path) - 1):
        edges_to_add.append((shortest_path[i], shortest_path[i + 1]))

    for i in range(len(edges_to_add)):
        edge = graph.es.find(_source=edges_to_add[i][0], _target=edges_to_add[i][1])
        graph.add_edge(edge.source, edge.target, weight=edge["weight"])

    return graph


def ga_solution(graph: igraph.Graph):
    # the most important GA params
    population_size = 30
    num_generations = 100
    crossover_probability = 0.9
    mutation_probability = 0.1

    # less important params
    num_parents_mating = int(population_size / 2)
    keep_elitism = 2
    keep_parents = -1

    # setup and run the GA
    matrix = OddVerticesPathMatrix(graph)

    num_genes = len(matrix.odd_vertices)
    initial_population = generate_random_permutations(num_genes, population_size)

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        crossover_probability=crossover_probability,
        crossover_type=crossover,
        mutation_probability=mutation_probability,
        mutation_percent_genes="default",
        mutation_type=mutate,
        fitness_func=lambda sol, index: fitness(matrix, sol, index),
        parent_selection_type="rws",  # roulette
        initial_population=initial_population,
        keep_elitism=keep_elitism,
        keep_parents=keep_parents,
    )

    ga_instance.run()

    paths, cost, phenotype = interpret_ga_solution(matrix, ga_instance.best_solution())

    # transform according to the solution
    euler_graph, new_edges = duplicate_edges_on_paths(graph, paths)

    #############################################################
    # output solution
    #############################################################
    print(f"phenotype={phenotype};\tcost={cost}")
    print(f"edges to add: {new_edges}")

    # plot the original and the transformed graph
    layout = graph.layout()
    fig, axs = pyplot.subplots(2, 1, figsize=(10, 10))

    visual_style = {
        "vertex_label_color": "blue",
        "vertex_label": range(graph.vcount()),
        # "vertex_color": "blue",
        "vertex_size": 0.3
    }

    igraph.drawing.plot(graph, layout=layout, target=axs[0], **visual_style)
    igraph.drawing.plot(euler_graph, layout=layout, target=axs[1], **visual_style)
    pyplot.show()
    return euler_graph


def main():
    #############################################################
    # find solution
    #############################################################

    # configure
    graph_file = "graph.dat"
    graph = read_graph_from_file(graph_file)

    if is_euler_graph(graph):
        fleury(graph)
    elif is_half_euler_graph(graph):
        fleury(fix_half_euler_graph(graph))
    else:
        euler_graph = ga_solution(graph)
        fleury(euler_graph)


if __name__ == "__main__":
    main()
