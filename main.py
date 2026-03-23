""" Run the program, ask the user questions, and show results interactively."""

from __future__ import annotations
import math
import data_loader
import recommendation
import visualizer
from graph_model import CompatibilityGraph


def get_user_input() -> list[float]:
    """Return a numerical feature vector based on the user's answers to terminal prompts.

    Preconditions:
        - user input must be valid floats for traits and int for love language

    >>> # Skipping doctest since it requires active stdin terminal user interaction.
    """
    print("\n--- Please rate the following traits based on your intuition (from 0 to 10) ---")
    o = float(input("Your openness to new things: ")) / 10
    c = float(input("Your level of organization and responsibility (Conscientiousness): ")) / 10
    e = float(input("Your social extroversion (Extraversion): ")) / 10
    a = float(input("Your affinity/teamwork ability (Agreeableness): ")) / 10
    s = float(input("Your emotional expression/sensitivity (Expressiveness): ")) / 10

    print("What's your love language? (1:Quality Time, 2:Physical Touch, 3:Acts of service, 4:Receiving Gifts, "
          "5:Word of Affirmation)")

    choice = 0
    while choice not in {1, 2, 3, 4, 5}:
        try:
            choice = int(input("Please select the number (1-5): "))
            if choice not in {1, 2, 3, 4, 5}:
                print("Invalid input! Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input! Please enter an integer.")

    love_lang_vec = [1 if i + 1 == choice else 0 for i in range(5)]
    user_vector = [o, c, e, a, s] + love_lang_vec
    return user_vector


def main() -> None:
    """Execute the main workflow of the Cupid Graphing recommendation system.

    >>> # Full workflow script execution cannot be doctested linearly due to terminal dependency.
    """
    print("Welcome to Cupid Interactive System!")
    print("\n[Step 1/4] Taking Personality Survey...")
    new_user_vector = get_user_input()
    new_user_id = "New_User_Interactive"
    print("\n[Step 2/4] Loading background datasets and establishing algorithms...")
    user_vectors = data_loader.generate_user_vectors("cupid_algorithm_dataset.csv")

    if user_vectors:
        sample_dim = len(list(user_vectors.values())[0])
        max_d = math.sqrt(sample_dim)
        # Pad dimension safely
        if len(new_user_vector) < sample_dim:
            new_user_vector.extend([0.0] * (sample_dim - len(new_user_vector)))
    else:
        max_d = 1.0

    sim_func = recommendation.SimilarityCalculator(max_d)

    print("[Step 3/4] Compiling social nodes and mapping Euclidean distances...")
    graph = CompatibilityGraph(threshold=0.75)
    graph.add_users_as_nodes(user_vectors)

    # 1. Connect background users
    graph.build_edges(sim_func)

    # 2. Add our new user
    graph.add_new_user(new_user_id, new_user_vector, sim_func)

    print("[Step 4/4] Graph built successfully! Running 2-degree Weighted BFS logic...")
    matches = recommendation.weighted_bfs_recommendation(graph.graph, new_user_id, top_n=5)

    print("\n" + "=" * 40)
    print("--- YOU HAVE BEEN MATCHED! ---")
    print("=" * 40)
    for rank, match in enumerate(matches, 1):
        print(f"#{rank}: {match[0]} (Compatibility Score: {match[1]:.4f})")

    print("\nRendering visualization graph in your browser... Please wait.")
    visualizer.draw_interactive_cluster(graph.graph, new_user_id, matches)


if __name__ == "__main__":
    # import python_ta
    #
    # python_ta.check_all(config={
    #     'extra-imports': ['data_loader', 'recommendation', 'visualizer', 'graph_model', 'math', 'typing'],
    #     'allowed-io': ['get_user_input', 'main'],
    #     'max-line-length': 120
    # })
    main()
