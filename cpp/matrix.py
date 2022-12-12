import igraph
import numpy


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