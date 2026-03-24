""" Draw the graph and highlight the best matches."""

from __future__ import annotations
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def draw_social_cluster(graph: nx.Graph, new_user_id: str, matches: list[tuple[str, float]]) -> None:
    """Render the graph visually using matplotlib.
    Highlight the new_user_id node and the nodes in the matches list using distinct colors.

    Preconditions:
        - new_user_id in graph
    """
    match_ids = {m[0] for m in matches}
    subgraph, pos = _build_layout_graph(graph, new_user_id, match_ids)

    color_map = []
    size_map = []
    for node in subgraph.nodes():
        if node == new_user_id:
            color_map.append('#ff3333')  # Red
            size_map.append(300)
        elif node in match_ids:
            color_map.append('#ffd700')  # Gold
            size_map.append(200)
        else:
            color_map.append('#a8d8ea')  # Light Blue
            size_map.append(50)

    plt.figure(figsize=(10, 8), facecolor='#f8f9fa')
    plt.title("Cupid Graphing Engine - Local Cluster", fontsize=16)

    nx.draw_networkx_edges(subgraph, pos, alpha=0.2, edge_color='#888', width=0.5)
    nx.draw_networkx_nodes(subgraph, pos, node_color="color_map",
                           node_size=size_map, edgecolors='#444', linewidths=1.0)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def _build_layout_graph(graph: nx.Graph, new_user_id: str, match_ids: set[str]) -> tuple[nx.Graph, dict]:
    """Helper to build subgraph and calculate spring layout positions."""
    nodes_to_draw = {new_user_id}
    nodes_to_draw.update(match_ids)

    for neighbor in graph.neighbors(new_user_id):
        if len(nodes_to_draw) > 200:
            break
        nodes_to_draw.add(neighbor)

    for node in graph.nodes():
        if len(nodes_to_draw) > 250:
            break
        nodes_to_draw.add(node)

    subgraph = graph.subgraph(nodes_to_draw).copy()
    layout_graph = subgraph.copy()
    for match_id in match_ids:
        if not layout_graph.has_edge(new_user_id, match_id):
            layout_graph.add_edge(new_user_id, match_id, weight=2.0)

    pos = nx.spring_layout(layout_graph, weight='weight', iterations=50)
    return subgraph, pos


def _get_standard_edges(subgraph: nx.Graph, pos: dict) -> go.Scatter:
    """Helper to create the standard edge trace."""
    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        edge_x.extend([pos[edge[0]][0], pos[edge[1]][0], None])
        edge_y.extend([pos[edge[0]][1], pos[edge[1]][1], None])

    return go.Scatter(
        x=edge_x, y=edge_y,
        line={"width": 0.5, "color": '#888'},
        hoverinfo='none',
        mode='lines'
    )


def _get_rec_edges(match_ids: set[str], new_user_id: str, pos: dict) -> go.Scatter:
    """Helper to create the recommendation edge trace."""
    rec_edge_x = []
    rec_edge_y = []
    for match_id in match_ids:
        if new_user_id in pos and match_id in pos:
            rec_edge_x.extend([pos[new_user_id][0], pos[match_id][0], None])
            rec_edge_y.extend([pos[new_user_id][1], pos[match_id][1], None])

    return go.Scatter(
        x=rec_edge_x, y=rec_edge_y,
        line={"width": 2, "color": 'gold', "dash": 'dot'},
        hoverinfo='none',
        mode='lines'
    )


def _get_node_trace(subgraph: nx.Graph, pos: dict, new_user_id: str,
                    match_ids: set[str], matches: list[tuple[str, float]]) -> list[go.Scatter]:
    """Helper to create separate node traces for User, Matches, and Regular nodes."""
    match_scores = dict(matches)

    # Store coordinate and hover text data for the three categories of nodes
    data = {
        'reg_x': [], 'reg_y': [], 'reg_text': [], 'reg_adj': [],
        'match_x': [], 'match_y': [], 'match_text': [],
        'user_x': [], 'user_y': [], 'user_text': []
    }

    # Populate the data dictionary based on node type
    for node in subgraph.nodes():
        x, y = pos[node]
        adj_count = len(list(subgraph.neighbors(node)))

        if node == new_user_id:
            data['user_x'].append(x)
            data['user_y'].append(y)
            data['user_text'].append(f"<b>You! ({node})</b><br># of connections: {adj_count}")
        elif node in match_ids:
            data['match_x'].append(x)
            data['match_y'].append(y)
            data['match_text'].append(
                f"<b>Top Match: {node}</b><br>Score: {match_scores[node]:.4f}<br>"
                f"# of connections: {adj_count}"
            )
        else:
            data['reg_x'].append(x)
            data['reg_y'].append(y)
            data['reg_text'].append(f"User: {node}<br># of connections: {adj_count}")
            data['reg_adj'].append(adj_count)

    # Build and return the scatter traces for the graph
    return [
        go.Scatter(
            x=data['reg_x'], y=data['reg_y'], mode='markers', hoverinfo='text', text=data['reg_text'],
            marker={
                "showscale": True, "colorscale": 'YlGnBu', "reversescale": True,
                "color": data['reg_adj'], "size": 10,
                "colorbar": {"thickness": 15, "title": {"text": 'Node Connections', "side": 'right'},
                             "xanchor": 'left'},
                "line_width": 1, "line_color": '#888'
            }
        ),
        go.Scatter(
            x=data['match_x'], y=data['match_y'], mode='markers', hoverinfo='text', text=data['match_text'],
            marker={"color": 'gold', "size": 20, "line_width": 2, "line_color": 'black'}
        ),
        go.Scatter(
            x=data['user_x'], y=data['user_y'], mode='markers', hoverinfo='text', text=data['user_text'],
            marker={"color": '#ff3333', "size": 30, "line_width": 2, "line_color": 'black'}
        )
    ]


def draw_interactive_cluster(graph: nx.Graph, new_user_id: str, matches: list[tuple[str, float]]) -> None:
    """Render a clean, ultra-premium interactive graph using Plotly.
    Users can hover to see the scores, zoom in/out, and explore the social cluster dynamics.

    Preconditions:
        - new_user_id in graph

    >>> import networkx as _nx
    >>> g = _nx.Graph()
    >>> g.add_node('U1'); g.add_node('U2')
    >>> # Browser rendering is strictly skipped in backend doctest logic.
    """
    match_ids = {m[0] for m in matches}
    subgraph, pos = _build_layout_graph(graph, new_user_id, match_ids)

    standard_edge_trace = _get_standard_edges(subgraph, pos)
    rec_edge_trace = _get_rec_edges(match_ids, new_user_id, pos)

    # Get the split node traces
    node_traces = _get_node_trace(subgraph, pos, new_user_id, match_ids, matches)

    layout = go.Layout(
        title={"text": "<br>Network graph made with Python", "font": {"size": 16}},
        showlegend=False,
        hovermode='closest',
        margin={"b": 20, "l": 5, "r": 5, "t": 40},
        plot_bgcolor='#e5ecf6',
        paper_bgcolor='#e5ecf6',
        annotations=[{
            "text": ("Python code: <a href='https://plotly.com/python/network-graphs/'> "
                     "https://plotly.com/python/network-graphs/</a>"),
            "showarrow": False, "xref": "paper", "yref": "paper",
            "x": 0.005, "y": -0.002
        }],
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False}
    )

    # Combine edge traces and all node traces
    fig = go.Figure(data=[standard_edge_trace, rec_edge_trace] + node_traces, layout=layout)
    fig.show(renderer="browser")


if __name__ == '__main__':
    pass
    # import python_ta
    # python_ta.check_all(config={
    #     'extra-imports': ['networkx', 'plotly.graph_objects', 'matplotlib.pyplot'],
    #     'allowed-io': ['draw_interactive_cluster', 'draw_social_cluster'],
    #     'max-line-length': 120
    # })
