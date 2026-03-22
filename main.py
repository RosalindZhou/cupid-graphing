# Run the program, ask the user questions, and show results.

from __future__ import annotations

import data_loader
import recommendation
import visualizer
from graph_model import CompatibilityGraph


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
    # 1. Lucy: Load the cleaned user data vector
    # The function "generate_user_vectors" should return a dictionary of the form {ID: [0.1, 0.5, ...]}
    user_vectors = data_loader.generate_user_vectors("cupid_algorithm_dataset.csv")

    # 2. Role A: Initialize the graph and add existing users as nodes
    # Set the threshold at 0.75. Only nodes with a similarity level exceeding this value will be connected by edges.
    graph = CompatibilityGraph(threshold=0.75)
    graph.add_users_as_nodes(user_vectors)

    # 3. Lucy: Construct the initial social network edges based on the similarity algorithm
    # Here, the Euclidean calculation function written by Rosalind is passed in as a parameter
    graph.build_edges(recommendation.calculate_euclidean_weight)

    # Obtain the new user data of the currently running program
    new_user_vector = get_user_input()
    new_user_id = "New_User_Interactive"

    # Insert the new user's activity into the chart
    graph.add_new_user(new_user_id, new_user_vector)

    # 5. Rosalind: Execute 2-degree Breadth-First Search to recommend logical patterns for finding the best match
    # Return the top 5 matching candidates along with their scores
    matches = recommendation.weighted_bfs_recommendation(graph.graph, new_user_id, top_n=5)

    # 6. Anna: Call the visual rendering results
    print(f"\nThe most suitable match we found for you is: {matches}")
    visualizer.draw_social_cluster(graph.graph, new_user_id, matches)


if __name__ == "__main__":
    # When you are ready to check your work with python_ta, uncomment the following lines.
    # import python_ta
    # python_ta.check_all(config={
    #     'max-line-length': 120,
    #     'disable': ['R1705', 'E9998', 'E9999', 'static_type_checker']
    # })

    main()
