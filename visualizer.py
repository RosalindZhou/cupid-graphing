# Draw the graph and highlight the best matches.

from __future__ import annotations
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go


def draw_social_cluster(graph: nx.Graph, new_user_id: str, matches: list[tuple[str, float]]) -> None:
    """Render the graph visually using matplotlib.
    Highlight the new_user_id node and the nodes in the matches list using distinct colors.
    
    Preconditions:
        - new_user_id in graph
    """
    pos = nx.spring_layout(graph, weight='weight')
    color_map = []
    match_ids = {m[0] for m in matches}

    for node in graph.nodes:
        if node == new_user_id:
            color_map.append('red')  # new user
        elif node in match_ids:
            color_map.append('gold')  # golden matcher
        else:
            color_map.append('lightgray')  # common user

    nx.draw(graph, pos, node_color=color_map, with_labels=False)
    plt.show()


def draw_premium_interactive_cluster(graph: nx.Graph, new_user_id: str, matches: list[tuple[str, float]]) -> None:
    """Render a clean, ultra-premium interactive graph using Plotly.
    Users can hover to see the scores, zoom in/out, and explore the social cluster dynamics.
    
    Preconditions:
        - new_user_id in graph
        
    >>> import networkx as nx
    >>> g = nx.Graph()
    >>> g.add_node('U1'); g.add_node('U2')
    >>> # Browser rendering is strictly skipped in backend doctest logic.
    """
    nodes_to_draw = {new_user_id}
    match_ids = {m[0] for m in matches}
    nodes_to_draw.update(match_ids)
    
    # Add direct connections to build a local cluster
    for neighbor in graph.neighbors(new_user_id):
        if len(nodes_to_draw) > 200:
            break
        nodes_to_draw.add(neighbor)
        
    for node in graph.nodes():
        if len(nodes_to_draw) > 250:
            break
        nodes_to_draw.add(node)
        
    subgraph = graph.subgraph(nodes_to_draw).copy()
    
    # --- UI LAYOUT FIX ---
    # To make the yellow matches appear physically closer to the User,
    # we temporarily add high-weight invisible edges to the layout algorithm!
    layout_graph = subgraph.copy()
    for match_id in match_ids:
        # Pull matches close to the user with a heavy weight
        if not layout_graph.has_edge(new_user_id, match_id):
            layout_graph.add_edge(new_user_id, match_id, weight=2.0)
            
    # Calculate positions based on the pulled layout
    pos = nx.spring_layout(layout_graph, weight='weight', iterations=50)

    # Prepare Standard Edges (Solid)
    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    standard_edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Prepare Recommendation Edges (Golden Dashed)
    rec_edge_x = []
    rec_edge_y = []
    for match_id in match_ids:
        if new_user_id in pos and match_id in pos:
            x0, y0 = pos[new_user_id]
            x1, y1 = pos[match_id]
            rec_edge_x.extend([x0, x1, None])
            rec_edge_y.extend([y0, y1, None])

    rec_edge_trace = go.Scatter(
        x=rec_edge_x, y=rec_edge_y,
        line=dict(width=2, color='gold', dash='dot'),
        hoverinfo='none',
        mode='lines')

    # Prepare Nodes
    node_x = []
    node_y = []
    hover_text = []
    node_color = []
    node_size = []
    match_scores = {m[0]: m[1] for m in matches}

    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        if node == new_user_id:
            node_color.append('#ff3333')  # Red
            node_size.append(22)
            hover_text.append(f"<b>You! ({node})</b><br>Hovering in the center of your universe.")
        elif node in match_ids:
            node_color.append('#ffd700')  # Gold
            node_size.append(18)
            hover_text.append(f"<b>Top Match: {node}</b><br>Compatibility Score: {match_scores[node]:.4f}")
        else:
            node_color.append('#a8d8ea')  # Light Blue
            node_size.append(10)
            hover_text.append(f"User: {node}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=hover_text,
        textposition="bottom center",
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(width=1, color='black')
        ))

    fig = go.Figure(data=[standard_edge_trace, rec_edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Cupid Graphing Engine - Enterprise Topology',
                        title_font_size=20,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=50),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        annotations=[dict(
                            text="<b>Red: You | Gold Dashed: AI Recommendations | Blue: Network Members</b>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show(renderer="browser")

if __name__ == '__main__':
    # Uncomment to check formatting
    # import python_ta
    # python_ta.check_all(config={
    #     'extra-imports': ['matplotlib.pyplot', 'networkx', 'plotly.graph_objects'],
    #     'allowed-io': ['draw_social_cluster', 'draw_interactive_plotly_cluster'],
    #     'max-line-length': 120
    # })
    pass
