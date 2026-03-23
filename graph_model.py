
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

        >>> g = CompatibilityGraph(threshold=0.5)
        >>> g.add_users_as_nodes({'A': [1.0], 'B': [0.8], 'C': [0.1]})
        >>> g.build_edges(lambda u, v: 1.0 - abs(u[0] - v[0]))
        >>> g.graph.has_edge('A', 'B')
        True
        >>> g.graph.has_edge('A', 'C')
        False
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
        """Add a new user dynamically and compute edges to existing nodes.
        Ensures the new user is connected to at least their 5 top matches (if possible)
        even if weights are strictly below the threshold.

        Preconditions:
            - new_user_id not in self.graph

        >>> g = CompatibilityGraph(threshold=0.9)
        >>> g.add_users_as_nodes({'A': [1.0], 'B': [1.0]})
        >>> g.add_new_user('C', [0.1], lambda u, v: 1.0 - abs(u[0] - v[0]))
        >>> g.graph.has_edge('C', 'A') # Forced connection due to top 5 fallback
        True
        """
        if new_user_id in self.graph:
            raise ValueError('new_user_id already exists in graph')
        existing_nodes = list(self.graph.nodes)
        self.graph.add_node(new_user_id, vector=new_user_vector)
        weights = []
        for existing_id in existing_nodes:
            existing_vector = self.graph.nodes[existing_id]['vector']
            weight = similarity_function(new_user_vector, existing_vector)
            weights.append((existing_id, weight))

        weights.sort(key=lambda x: x[1], reverse=True)

        connected = 0
        for existing_id, weight in weights:
            # Force connect to at least top 5 users if threshold is too strict
            if weight > self.threshold or connected < 5:
                self.graph.add_edge(new_user_id, existing_id, weight=weight)
                connected += 1


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
