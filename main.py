# Run the program, ask the user questions, and show results.

from data_loader import generate_user_vectors
from graph_model import CompatibilityGraph
from recommendation import calculate_euclidean_weight, weighted_bfs_recommendation
from visualizer import draw_social_cluster

def get_user_input():
    """Ask the user questions in the terminal and return their answers as a vector.
    """

def main():
    print("Starting Cupid Graphing Engine...")
    
    # Load data
    user_data = generate_user_vectors('cupid_algorithm.csv')
    
    # Build graph
    # TODO
    social_graph = CompatibilityGraph(threshold=)
    social_graph.add_users_as_nodes(user_data)
    social_graph.build_edges(calculate_euclidean_weight)
    
    # Get new user info
    new_user = get_user_input()
    
    # Add user and find matches
    social_graph.add_new_user('NEW_USER_01', new_user)
    top_matches = weighted_bfs_recommendation(social_graph.graph, 'NEW_USER_01', top_n=5)
    
    print(f"Your top matches are: {top_matches}")
    
    # 5. Show the graph
    draw_social_cluster(social_graph.graph, 'NEW_USER_01', top_matches)

if __name__ == "__main__":
    main()
