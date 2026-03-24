""" Run the program, ask the user questions, and show results interactively."""

from __future__ import annotations
import math
from typing import Any

import data_loader
import recommendation
import visualizer
from graph_model import CompatibilityGraph


def get_valid_input(prompt, input_type=Any, min_val=None, max_val=None, options=None):
    """
    Universal Input Validator
    :param prompt: Prompt text
    :param input_type: Expected type (int, float, str)
    :param min_val: Minimum value (applicable only to numbers)
    :param max_val: Maximum value (applicable only to numbers)
    :param options: Valid selection list (for strings or fixed options)
    """
    while True:
        user_input = input(prompt).strip()

        if not user_input:
            print("Error: Input cannot be empty. Please try again.")
            continue

        try:
            value = input_type(user_input)

            if input_type in [int, float]:
                if min_val is not None and value < min_val:
                    print(f"Error: Value too small. Min allowed is {min_val}.")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Error: Value too large. Max allowed is {max_val}.")
                    continue

            # check Location or Love Language
            if options is not None and value not in options:
                print(f"Error: Invalid choice. Please choose from {options}.")
                continue

            return value

        except ValueError:
            print(f"Error: Invalid format. Please enter a valid {input_type.__name__}.")


def get_user_input() -> dict:
    """Return a numerical feature vector based on the user's answers to terminal prompts.

    Preconditions:
        - user input must be valid floats for traits and int for love language

    >>> # Skipping doctest since it requires active stdin terminal user interaction.
    """
    print("\n" + "=" * 50)
    print("--- CUPID MATCHING SYSTEM: PERSONAL PROFILE SURVEY ---")
    print("--- (All fields are mandatory and validated) ---")
    print("=" * 50)

    # 1-4.
    age = get_valid_input("1. Age (18-100): ", int, 18, 100)
    edu = get_valid_input("2. Education (1:University/College, 2:Bachelor, 3:PhD): ", int, 1, 5)
    loc_options = ["Urban", "Suburban", "Rural"]
    loc_msg = f"3. Select Location ({'/'.join(loc_options)}): "
    loc = get_valid_input(loc_msg, str, options=loc_options)
    field = get_valid_input("4. Career Field (e.g., IT, Art, Law): ", str).strip().title()

    # 5-11.
    print("\n--- Please rate the following (0 to 10) ---")
    ambition = get_valid_input("5. Career Ambition: ", float, 0, 10) / 10
    o = get_valid_input("6. Openness to new things: ", float, 0, 10) / 10
    e = get_valid_input("7. Social extroversion (Extraversion): ", float, 0, 10) / 10
    a = get_valid_input("8. Teamwork ability (Agreeableness): ", float, 0, 10) / 10
    c = get_valid_input("9. Organization/Responsibility (Conscientiousness): ", float, 0, 10) / 10
    chronotype = get_valid_input("10. Chronotype (0: Morning bird, 10: Night owl): ", float, 0, 10) / 10
    spontaneity = get_valid_input("11. Spontaneity level: ", float, 0, 10) / 10

    # 12. love language
    print("\n12. What is your Love Language?")
    love_lang_map = {
        1: "Quality Time", 2: "Physical Touch", 3: "Acts of Service",
        4: "Receiving Gifts", 5: "Words of Affirmation"
    }
    for k, v in love_lang_map.items():
        print(f"  {k}: {v}")

    choice = get_valid_input("Select (1-5): ", int, 1, 5)

    # 13. emotion expression
    expressive = get_valid_input("13. Emotional Expressiveness (0-10): ", float, 0, 10) / 10

    return {
        'age': age, 'education': edu, 'location': loc, 'career_field': field,
        'career_ambition': ambition, 'openness': o, 'extraversion': e,
        'agreeableness': a, 'conscientiousness': c, 'chronotype': chronotype,
        'spontaneity': spontaneity, 'love_language': love_lang_map[choice],
        'emotional_expressiveness': expressive
    }


def main() -> None:
    """Execute the main workflow of the Cupid Graphing recommendation system.

    >>> # Full workflow script execution cannot be doctested linearly due to terminal dependency.
    """
    print("Welcome to Cupid Interactive System!")

    # [Step 1] Obtain the original dictionary of new users
    raw_new_user = get_user_input()
    new_user_id = "New_User_Interactive"

    # [Step 2] Load the background data and perform dynamic alignment
    print("\n[Step 2/4] Loading and aligning datasets...")
    bg_data = data_loader.load_and_clean_data("cupid_algorithm_dataset.csv")

    combined_data = bg_data + [raw_new_user]

    # Call the cleaning process of the data_loader to perform a unified conversion
    for col in data_loader.CATEGORICAL_COLUMNS:
        combined_data = data_loader.apply_one_hot_encoding(combined_data, col)
    combined_data = data_loader.normalize_numerical_data(combined_data, data_loader.NUMERICAL_COLUMNS)

    # Determine the final order of the feature columns
    feature_columns = data_loader.NUMERICAL_COLUMNS.copy()
    all_keys = combined_data[0].keys()
    encoded_cols = sorted([k for k in all_keys if k not in feature_columns and k != 'user_id'])
    feature_columns.extend(encoded_cols)

    # Separate out the background user vector and the new user vector
    user_vectors = {}
    for i in range(len(combined_data) - 1):
        row = combined_data[i]
        user_vectors[row['user_id']] = [float(row[col]) for col in feature_columns]

    new_user_vector = [float(combined_data[-1][col]) for col in feature_columns]

    # [Step 3] Build a graph model
    sample_dim = len(feature_columns)
    max_d = math.sqrt(sample_dim)
    sim_func = recommendation.SimilarityCalculator(max_d)

    print("[Step 3/4] Building Compatibility Graph...")
    graph = CompatibilityGraph(threshold=0.75)
    graph.add_users_as_nodes(user_vectors)
    graph.build_edges(sim_func)
    graph.add_new_user(new_user_id, new_user_vector, sim_func)

    # [Step 4] Run 2-degree Breadth-First Search recommendation
    print("[Step 4/4] Finding matches via transitive similarity...")
    matches = recommendation.weighted_bfs_recommendation(graph.graph, new_user_id, top_n=5)

    print("\n" + "=" * 40 + "\n--- MATCHES FOUND ---" + "\n" + "=" * 40)
    for rank, (m_id, score) in enumerate(matches, 1):
        print(f"#{rank}: {m_id} (Compatibility Score: {score:.4f})")

    # Visualization
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
