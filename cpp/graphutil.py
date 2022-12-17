from enum import Enum
import random
from typing import Optional
import igraph
import numpy


class GraphType(str, Enum):
    generic = "generic"
    half_euler = "half_euler"
    euler = "euler"

    @staticmethod
    def of(graph: igraph.Graph):
        if is_euler_graph(graph):
            return GraphType.euler
        elif is_half_euler_graph(graph):
            return GraphType.half_euler
        else:
            return GraphType.generic


# todo: optimize evaluation (see Floyd-Warshall's algorithm)
class PathMatrix:
    def __init__(self, graph: igraph.Graph, output="epath", vertices: list[int] | None = None):
        if vertices is None:
            vertices = list(range(graph.vcount()))
        self.__vertices = vertices
        num_vertices = len(self.__vertices)

        self.__min_paths_costs = numpy.ndarray(shape=(num_vertices, num_vertices), dtype=int)
        self.__min_paths = numpy.ndarray(shape=(num_vertices, num_vertices), dtype=list)

        # for i, v in enumerate(self.__vertices):
        #     paths = graph.get_shortest_paths(
        #         v,
        #         to=self.__vertices[:i+1],
        #         weights=graph.es["weight"],
        #         output="epath"
        #     )
        #     self.__min_paths.append(paths)
        #
        #     distances = numpy.ndarray(shape=(1, num_vertices))
        #     for j, path in enumerate(paths):
        #         distance = 0
        #         for edge in path:
        #             distance += graph.es[edge]["weight"]
        #         distances[j] = distance
        #
        #     self.__min_paths_costs[v][:] = distances

        # self.__min_paths_costs += numpy.transpose(self.__min_paths_costs)

        # evaluate path cost matrix
        for i, j in numpy.ndindex(self.__min_paths_costs.shape):
            paths = graph.get_shortest_paths(
                self.__vertices[i],
                to=self.__vertices[j],
                weights=graph.es["weight"],
                output=output
            )
            path = paths[0]
            distance = 0
            for edge in path:
                distance += graph.es[edge]["weight"]

            self.__min_paths_costs[i, j] = distance
            self.__min_paths[i, j] = path

    @property
    def min_paths_costs(self):
        return self.__min_paths_costs

    @property
    def min_paths(self):
        return self.__min_paths

    @property
    def vertices(self):
        return self.__vertices


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
    edge_list = graph.get_edgelist()
    res = graph.bridges()
    bridges: list[tuple[int, int]] = list(map(lambda x: edge_list[x], res))

    for bridge in bridges:
        if (bridge[0] == edge[0] and bridge[1] == edge[1]) or (bridge[0] == edge[1] and bridge[1] == edge[0]):
            return True
    return False


def choose_edge(graph: igraph.Graph, visited_edges: list[tuple[int, int]], vertex: int) -> tuple[int, int]:
    copy: igraph.Graph = graph.copy()
    for i in range(len(visited_edges)):
        edge_to_delete: Optional[igraph.Edge] = copy.es.find(_source=visited_edges[i][0], _target=visited_edges[i][1])
        copy.delete_edges(edge_to_delete)

    all_possible_edges = list(map(lambda x: (vertex, x), graph.get_adjlist()[vertex]))
    available_non_visited_edges = list(map(lambda x: (vertex, x), copy.get_adjlist()[vertex]))
    available_non_bridges = list(filter(lambda x: (not is_bridge(copy, x)), available_non_visited_edges))

    if len(available_non_bridges) > 0:
        return available_non_bridges[0]
    elif len(available_non_visited_edges) > 0:
        return available_non_visited_edges[0]
    else:
        return all_possible_edges[0]


def is_graph_valid_for_fleury(graph: igraph.Graph):
    assert (graph.is_connected())
    for vertex in graph.get_adjacency():
        assert (sum(vertex) % 2 == 0)


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
    is_graph_valid_for_fleury(graph)
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


def create_random_graph(vertices: int, edge_probabilty: float, max_weight: int):
    edges = []

    for begin_vertex in range(0, vertices):
        edge_count = 0
        for end_vertex in range(0, vertices):
            if edge_count > 2:
                break
            if begin_vertex == end_vertex:
                continue
            else:
                prob = random.random()
                if prob <= edge_probabilty:
                    edge_count += 1
                    weight = random.choice(range(1, max_weight))
                    edges.append([begin_vertex, end_vertex, weight])

    return edges


def generate_random_permutations(len_permutation, num_permutations):
    population = [list(range(len_permutation)) for _ in range(num_permutations)]
    for solution in population:
        random.shuffle(solution)
    return population
