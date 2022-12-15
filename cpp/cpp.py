from __future__ import annotations

from enum import Enum
import igraph
import pygad


from cpp.ga import create_template_ga_instance, find_euler_transform_by_ga
from cpp.graphutil import GraphType, fix_half_euler_graph, duplicate_edges_on_paths, fleury
from cpp.matrix import OddVerticesPathMatrix


def transform_to_euler_by_ga(graph: igraph.Graph, ga_instance: pygad.GA | None = None):
    matrix = OddVerticesPathMatrix(graph)
    if ga_instance is None:
        ga_instance = create_template_ga_instance(matrix)
    paths, cost, _ = find_euler_transform_by_ga(matrix, ga_instance)
    euler_graph, _ = duplicate_edges_on_paths(graph, paths)
    return euler_graph, cost


def auto_solve_cpp(graph: igraph.Graph, ga_instance: pygad.GA | None = None):
    cost = 0
    graph_type = GraphType.of(graph)
    if graph_type is GraphType.euler:
        euler_graph = graph
    elif graph_type is GraphType.half_euler:
        euler_graph = fix_half_euler_graph(graph)
    else:
        euler_graph, cost = transform_to_euler_by_ga(graph, ga_instance)
    path_vertices = fleury(euler_graph)
    return path_vertices, euler_graph, graph_type, cost

