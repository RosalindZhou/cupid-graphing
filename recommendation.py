""" Calculate distances, similarity weights, and run Weighted BFS."""

from __future__ import annotations
import math
import networkx as nx


def calculate_euclidean_weight(vec_u: list[float], vec_v: list[float], max_d: float) -> float:
    """Return a similarity weight between 0.0 and 1.0 based on the Euclidean distance
    between vec_u and vec_v.

    Preconditions:
    - len(vec_u) == len(vec_v)
    - max_d > 0.0
    >>> calculate_euclidean_weight([0.0, 0.0], [3.0, 4.0], 10.0)
    0.5
    >>> calculate_euclidean_weight([1.0, 1.0], [1.0, 1.0], 5.0)
    1.0
    """
    distance = math.dist(vec_u, vec_v)
    return max(0.0, 1.0 - (distance / max_d))


class SimilarityCalculator:
    """
    A callable class that calculates Euclidean weight
    using a predefined max_d.

    Instance Attributes:
        - max_d: The maximum distance used for scaling the similarity.

    Representation Invariants:
        - self.max_d > 0.0

    >>> sim = SimilarityCalculator(10.0)
    >>> sim([0.0, 0.0], [3.0, 4.0])
    0.5
    """
    max_d: float

    def __init__(self, max_d: float) -> None:
        """Initialize the SimilarityCalculator with a maximum distance.
        
        >>> sim = SimilarityCalculator(10.0)
        >>> sim.max_d
        10.0
        """
        self.max_d = max_d

    def __call__(self, u: list[float], v: list[float]) -> float:
        """Calculate the similarity weight between two vectors u and v.
        
        >>> sim = SimilarityCalculator(10.0)
        >>> sim([0.0, 0.0], [3.0, 4.0])
        0.5
        """
        return calculate_euclidean_weight(u, v, self.max_d)


def dot_product_similarity(vec_u: list[float], vec_v: list[float]) -> float:
    """Return a similarity score based on the dot product (angle) of vec_u and vec_v.
    Preconditions:
        - len(vec_u) == len(vec_v)

    Returns the cosine similarity between vec_u and vec_v.

    >>> round(dot_product_similarity([1.0, 0.0], [0.0, 1.0]), 5)
    0.0
    >>> round(dot_product_similarity([1.0, 1.0], [2.0, 2.0]), 5)
    1.0
    """
    dot_product = sum(u * v for u, v in zip(vec_u, vec_v))
    mag_u = math.sqrt(sum(u ** 2 for u in vec_u))
    mag_v = math.sqrt(sum(v ** 2 for v in vec_v))
    if mag_u == 0.0 or mag_v == 0.0:
        return 0.0
    return dot_product / (mag_u * mag_v)


def weighted_bfs_recommendation(graph: nx.Graph, start_node: str, top_n: int = 5) -> list[tuple[str, float]]:
    """Return a list of the top_n recommended users for the start_node by finding
    2-degree connections (friends of friends).

    The score is calculated as $Score(A,C) = w_1 \times w_2$.

    Preconditions:
        - top_n > 0
        - start_node != ''

    >>> import networkx as nx
    >>> g = nx.Graph()
    >>> g.add_edge('A', 'B', weight=0.5)
    >>> g.add_edge('B', 'C', weight=0.5)
    >>> weighted_bfs_recommendation(g, 'A', 5)
    [('C', 0.25)]
    """
    # If the starting user is not in the graph, we cannot recommend anyone.
    if start_node not in graph:
        return []

    # Level 1: Get all direct friends of the starting user.
    direct_friends = set(graph.neighbors(start_node))

    # Dictionary to keep track of the highest recommendation score for each friend-of-friend.
    recommendations = {}

    # Iterate through each direct friend to find their connections (Level 2).
    for friend in direct_friends:
        # w1 is the compatibility weight between the start_node and the direct friend.
        w1 = graph[start_node][friend]['weight']

        # Now, look at the friends of our friend (fof).
        for fof in graph.neighbors(friend):
            # A valid friend-of-friend should not be ourselves, and not already our direct friend.
            if fof != start_node and fof not in direct_friends:
                # w2 is the compatibility weight between our friend and their friend.
                w2 = graph[friend][fof]['weight']

                # The indirect score is the product of both edge weights.
                score = w1 * w2

                # If we haven't seen this fof before, add them to our recommendations.
                if fof not in recommendations:
                    recommendations[fof] = score
                else:
                    # If we found another path to this same fof, keep the maximum path score.
                    recommendations[fof] = max(recommendations[fof], score)

    # Sort the dictionary items based on the score in descending order (highest first).
    sorted_recommendations = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)

    # Return only the top N results as a list of tuples containing (user_id, score).
    return sorted_recommendations[:top_n]


def explore_indirect_vs_direct(graph: nx.Graph, node_a: str, node_c: str, indirect_score: float) -> None:
    """Print a comparison between the calculated indirect_score and the direct
    compatibility weight between node_a and node_c.

    Preconditions:
        - node_a != '' and node_c != ''
        - indirect_score >= 0.0

    >>> import networkx as nx
    >>> g = nx.Graph()
    >>> g.add_node('A', vector=[0.0, 0.0, 0.0, 0.0])
    >>> g.add_node('C', vector=[1.0, 0.0, 0.0, 0.0])
    >>> explore_indirect_vs_direct(g, 'A', 'C', 0.8)
    Comparison between A and C:
      Indirect Score (BFS): 0.8000
      Direct Score (Euclidean): 0.5000
      Difference: 0.3000
    """
    if node_a not in graph or node_c not in graph:
        print("Cannot compare: one or both nodes missing.")
        return

    vec_a = graph.nodes[node_a]['vector']
    vec_c = graph.nodes[node_c]['vector']

    # max_d is assumed to be sqrt(dimension) since vectors are normalized
    max_d = math.sqrt(len(vec_a))
    direct_score = calculate_euclidean_weight(vec_a, vec_c, max_d)

    print(f"Comparison between {node_a} and {node_c}:")
    print(f"  Indirect Score (BFS): {indirect_score:.4f}")
    print(f"  Direct Score (Euclidean): {direct_score:.4f}")
    print(f"  Difference: {indirect_score - direct_score:.4f}")


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['math', 'networkx'],
        'allowed-io': ['explore_indirect_vs_direct'],
        'max-line-length': 120
    })
