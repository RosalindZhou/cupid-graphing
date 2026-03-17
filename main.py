# Run the program, ask the user questions, and show results.

from __future__ import annotations
from data_loader import generate_user_vectors
from graph_model import CompatibilityGraph
from recommendation import calculate_euclidean_weight, weighted_bfs_recommendation
from visualizer import draw_social_cluster

def get_user_input() -> list[float]:
    """Return a numerical feature vector based on the user's answers to terminal prompts.
    """

def main() -> None:
    """Execute the main workflow of the Cupid Graphing recommendation system."""
    print("Starting Cupid Graphing Engine...")
    
    # TODO: Add your main workflow logic here (load data, build graph, get input, draw graph)

if __name__ == "__main__":
    # When you are ready to check your work with python_ta, uncomment the following lines.
    # import python_ta
    # python_ta.check_all(config={
    #     'max-line-length': 120,
    #     'disable': ['R1705', 'E9998', 'E9999', 'static_type_checker']
    # })
    
    main()
