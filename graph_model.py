# This Python module contains the class that represents the social network graph.

"""Graph model for compatibility network."""
from __future__ import annotations
from typing import Callable
import networkx as nx


class CompatibilityGraph:
    """A graph class storing all user nodes and their compatibility edges.

    Instance Attributes:
        - graph: the networkx Graph object storing the network topology
        - threshold: the minimum similarity score required to create an edge

    Representation Invariants:
        - 0.0 <= self.threshold <= 1.0
    """
    graph: nx.Graph
    threshold: float

    def __init__(self, threshold: float = 0.75) -> None:
        """Initialize a new empty compatibility graph with the given threshold.

        >>> g = CompatibilityGraph(0.8)
        >>> g.threshold
        0.8
        >>> len(g.graph.nodes)
        0
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError('Threshold must be between 0.0 and 1.0.')
        self.graph = nx.Graph()
        self.threshold = threshold

    def add_users_as_nodes(self, user_vectors: dict[str, list[float]]) -> None:
        """Add all users from the user_vectors dictionary as vertices in the graph.

        Preconditions:
            - all(len(vec) > 0 for vec in user_vectors.values())

        >>> g = CompatibilityGraph()
        >>> g.add_users_as_nodes({'A': [0.1, 0.2], 'B': [0.3, 0.4]})
        >>> sorted(g.graph.nodes)
        ['A', 'B']
        >>> g.graph.nodes['A']['vector']
        [0.1, 0.2]
        """
        for user_id, vector in user_vectors.items():
            self.graph.add_node(user_id, vector=vector)

    def build_edges(self,
                    similarity_function: Callable[[list[float], list[float]], float]) -> None:
        """Calculate the weight between all pairs of nodes using the similarity_function.

        If the calculated weight exceeds self.threshold, add a weighted edge to the graph.
        
        Preconditions:
            - len(self.graph.nodes) > 0
        """
        nodes = list(self.graph.nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u = nodes[i]
                v = nodes[j]
                vector_u = self.graph.nodes[u]['vector']
                vector_v = self.graph.nodes[v]['vector']
                weight = similarity_function(vector_u, vector_v)
                if weight > self.threshold:
                    self.graph.add_edge(u, v, weight=weight)

    def add_new_user(self,
                     new_user_id: str,
                     new_user_vector: list[float],
                     similarity_function: Callable[[list[float], list[float]], float]) -> None:
        """Add a new user dynamically and compute edges to existing nodes."""
        if new_user_id in self.graph:
            raise ValueError('new_user_id already exists in graph')
        existing_nodes = list(self.graph.nodes)
        self.graph.add_node(new_user_id, vector=new_user_vector)
        for node in existing_nodes:
            existing_vector = self.graph.nodes[node]['vector']
            weight = similarity_function(new_user_vector, existing_vector)
            if weight > self.threshold:
                self.graph.add_edge(new_user_id, node, weight=weight)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    # Uncomment when ready
    # import python_ta
    # python_ta.check_all(config={
    #     'extra-imports': ['networkx'],
    #     'allowed-io': [],
    #     'max-line-length': 120
    # })
