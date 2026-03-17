# Draw the graph and highlight the best matches.

from __future__ import annotations
import matplotlib.pyplot as plt
import networkx as nx

def draw_social_cluster(graph: nx.Graph, new_user_id: str, matches: list[tuple[str, float]]) -> None:
    """Render the graph visually using matplotlib.
    Highlight the new_user_id node and the nodes in the matches list using distinct colors.
    """
