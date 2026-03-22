# This Python module contains the class that represents the social network graph.

from __future__ import annotations
import networkx as nx
from typing import Callable


class CompatibilityGraph:
    """A graph class storing all user nodes and their compatibility edges.

    Instance Attributes:
        - graph: the networkx Graph object storing the network topology
        - threshold: the minimum similarity score required to create an edge

    Representation Invariants:
        - self.threshold >= 0.0
        - self.threshold <= 1.0
    """
    graph: nx.Graph
    threshold: float

    # set the threshold, implement the methods and doctests
    def __init__(self, threshold: float = 0.75) -> None:
        """Initialize a new empty compatibility graph with the given threshold."""
        self.graph = nx.Graph()
        self.threshold = threshold

    def add_users_as_nodes(self, user_vectors: dict[str, list[float]]) -> None:
        """Add all users from the user_vectors dictionary as vertices in the graph.
        """

    def build_edges(self, similarity_function: Callable[[list[float], list[float]], float]) -> None:
        """Calculate the weight between all pairs of nodes using the similarity_function.
        If the calculated weight > self.threshold, add a weighted edge to the graph.
        """

    def add_new_user(self, new_user_id: str, new_user_vector: list[float]) -> None:
        """Add a new user to the graph dynamically and compute their edges with existing nodes.
        """
