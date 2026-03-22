# Run the program, ask the user questions, and show results interactively.

from __future__ import annotations
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

    choice = int(input("Please select the number: "))
    love_lang_vec = [1 if i + 1 == choice else 0 for i in range(5)]
    user_vector = [o, c, e, a, s] + love_lang_vec
    return user_vector

def create_similarity_function(max_d: float):
    """
    Creates a similarity function (closure) that calculates Euclidean weight
    using a predefined max_d.
    
    Preconditions:
        - max_d > 0.0
        
    >>> sim = create_similarity_function(10.0)
    >>> sim([0.0, 0.0], [3.0, 4.0])
    0.5
    """
    def similarity_func(u, v):
        return recommendation.calculate_euclidean_weight(u, v, max_d)
    return similarity_func


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
        import math
        max_d = math.sqrt(sample_dim)
        
        # Pad dimension safely 
        if len(new_user_vector) < sample_dim:
            new_user_vector.extend([0.0] * (sample_dim - len(new_user_vector)))
    else:
        max_d = 1.0

    sim_func = create_similarity_function(max_d)

    print("[Step 3/4] Compiling social nodes and mapping Euclidean distances...")
    graph = CompatibilityGraph(threshold=0.75)
    graph.add_users_as_nodes(user_vectors)
    
    # 1. Connect background users 
    graph.build_edges(sim_func)
    
    # 2. Add our new user
    graph.add_new_user(new_user_id, new_user_vector, sim_func)
    
    print("[Step 4/4] Graph built successfully! Running 2-degree Weighted BFS logic...")
    matches = recommendation.weighted_bfs_recommendation(graph.graph, new_user_id, top_n=5)
    
    print("\n" + "="*40)
    print("--- YOU HAVE BEEN MATCHED! ---")
    print("="*40)
    for rank, match in enumerate(matches, 1):
        print(f"#{rank}: {match[0]} (Compatibility Score: {match[1]:.4f})")
        
    if matches:
        print("\n--- Math Difference Exploration ---")
        top_match_id = matches[0][0]
        recommendation.explore_indirect_vs_direct(graph.graph, new_user_id, top_match_id, matches[0][1])
        
    print("\nRendering visualization graph in your browser... Please wait.")
    visualizer.draw_premium_interactive_cluster(graph.graph, new_user_id, matches)


if __name__ == "__main__":
    # When you are ready to check your work with python_ta, uncomment the following lines.
    # import python_ta
    # python_ta.check_all(config={
    #     'max-line-length': 120,
    #     'disable': ['R1705', 'E9998', 'E9999', 'static_type_checker']
    # })

    main()
