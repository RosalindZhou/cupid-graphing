# Draw the graph and highlight the best matches.

from __future__ import annotations
import matplotlib.pyplot as plt
import networkx as nx


def draw_social_cluster(graph: nx.Graph, new_user_id: str, matches: list[tuple[str, float]]) -> None:
    """Render the graph visually using matplotlib.
    Highlight the new_user_id node and the nodes in the matches list using distinct colors.
    
    Preconditions:
        - new_user_id in graph
    """
    pos = nx.spring_layout(graph, weight='weight')
    color_map = []
    match_ids = [m[0] for m in matches]

    for node in graph.nodes:
        if node == new_user_id:
            color_map.append('red')  # new user
        elif node in match_ids:
            color_map.append('gold')  # golden matcher
        else:
            color_map.append('lightgray')  # common user

    nx.draw(graph, pos, node_color=color_map, with_labels=False)
    plt.show()

if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['matplotlib.pyplot', 'networkx'],
        'allowed-io': ['draw_social_cluster'],
        'max-line-length': 120,
        'disable': ['E9998']
    })
