# Draw the graph and highlight the best matches.

import matplotlib.pyplot as plt
import networkx as nx

def draw_social_cluster(graph, new_user_id, matches):
    """Draw the graph using networkx.draw_spring(), highlight the new_user_id and their matches
    """
