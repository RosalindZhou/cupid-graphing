# Run the program, ask the user questions, and show results.

from __future__ import annotations

import data_loader
import recommendation
import visualizer
# from data_loader import generate_user_vectors
# from graph_model import CompatibilityGraph
# from recommendation import calculate_euclidean_weight, weighted_bfs_recommendation
# from visualizer import draw_social_cluster


def get_user_input() -> list[float]:
    """Return a numerical feature vector based on the user's answers to terminal prompts.
    """
    print("\n--- Welcome to the Cupid Graphing Matching System ---")
    print('\n--- Please rate the following traits based on your intuition (from 0 to 10) ---')
    o = float(input("Your openness to new things: ")) / 10
    c = float(input("Your level of organization and responsibility (Conscientiousness): ")) / 10
    e = float(input("Your social extroversion (Extraversion): ")) / 10
    a = float(input("Your affinity/teamwork ability (Agreeableness): ")) / 10
    s = float(input("Your emotional expression/sensitivity (Expressiveness): ")) / 10

    print("What's your love language? (1:Quality Time, 2:Physical Touch, 3:Acts of service, 4:Physical Touch, "
          "5:Word of Affirmation)")

    choice = int(input("Please select the number: "))
    love_lang_vec = [1 if i + 1 == choice else 0 for i in range(5)]
    user_vector = [o, e, a, c, s] + love_lang_vec
    return user_vector


def main() -> None:
    """Execute the main workflow of the Cupid Graphing recommendation system."""
    print("Starting Cupid Graphing Engine...")




if __name__ == "__main__":
    # When you are ready to check your work with python_ta, uncomment the following lines.
    # import python_ta
    # python_ta.check_all(config={
    #     'max-line-length': 120,
    #     'disable': ['R1705', 'E9998', 'E9999', 'static_type_checker']
    # })

    main()
