# Calculate distances, similarity weights, and run Weighted BFS.

from __future__ import annotations
import math
import networkx as nx


# TODO: implement functions and write doctest
def calculate_euclidean_weight(vec_u: list[float], vec_v: list[float], max_d: float) -> float:
    """Return a similarity weight between 0.0 and 1.0 based on the Euclidean distance
    between vec_u and vec_v.

    Preconditions:
    - len(vec_u) == len(vec_v)
    """


def calculate_dot_product_similarity(vec_u: list[float], vec_v: list[float]) -> float:
    """Return a similarity score based on the dot product (angle) of vec_u and vec_v.
    This is used to compare with the Euclidean method.
    """


def weighted_bfs_recommendation(graph: nx.Graph, start_node: str, top_n: int = 5) -> list[tuple[str, float]]:
    """Return a list of the top_n recommended users for the start_node by finding
    2-degree connections (friends of friends).

    The score is calculated as $Score(A,C) = w_1 \times w_2$.
    """


def explore_indirect_vs_direct(graph: nx.Graph, node_a: str, node_c: str, indirect_score: float) -> None:
    """Print a comparison between the calculated indirect_score and the direct
    compatibility weight between node_a and node_c.
    """
