""" Run the program, ask the user questions, and show results interactively."""

from __future__ import annotations
import math
from typing import Any

import data_loader
import recommendation
import visualizer
from graph_model import CompatibilityGraph


def get_demographics() -> tuple[int, int]:
    """Ask for and return (age, education).

    >>> # Skipping doctest since it requires active stdin terminal user interaction.
    """
    # 1. Age
    age = 0
    while age < 18:
        try:
            age = int(input("1. Enter your age (18+): "))
            if age < 18:
                print("Error: You must be at least 18 years old.")
        except ValueError:
            print("Error: Must enter a specific integer value for age.")

    # 2. Education
    print("\n2. Enter your education level:")
    print("1 (High School), 2 (Associate's), 3 (Bachelor's), 4 (Master's), 5 (Doctorate)")
    edu = 0
    while edu not in {1, 2, 3, 4, 5}:
        try:
            edu = int(input("Please select (1-5): "))
            if edu not in {1, 2, 3, 4, 5}:
                print("Error: Must enter a specific value from 1 to 5.")
        except ValueError:
            print("Error: Must enter a specific integer value (1-5).")
    return age, edu


def get_career_info() -> tuple[str, float]:
    """Ask for and return (career_field, career_ambition).

    >>> # Skipping doctest since it requires active stdin terminal user interaction.
    """
    careers = [
        "Tech", "Healthcare", "Finance", "Education", "Creative Arts",
        "Law", "Engineering", "Marketing", "Science", "Entrepreneurship"
    ]
    print("\n3. Enter your career field:")
    for idx, field_name in enumerate(careers, 1):
        print(f"{idx}: {field_name}")

    career_field = ""
    while career_field not in careers:
        try:
            field_choice = int(input(f"Please select (1-{len(careers)}): "))
            if 1 <= field_choice <= len(careers):
                career_field = careers[field_choice - 1]
            else:
                print(f"Error: Must enter a specific value between 1 and {len(careers)}.")
        except ValueError:
            print("Error: Must enter a specific integer value.")

    # 4. Career Ambition
    ambition = -1.0
    while not 0 <= ambition <= 10:
        try:
            ambition = float(input("\n4. Enter your career ambition level (0 to 10): "))
            if not 0 <= ambition <= 10:
                print("Error: Must enter a specific value from 0 to 10.")
        except ValueError:
            print("Error: Must enter a specific numerical value (0-10).")
    return career_field, ambition / 10


def get_personality_traits() -> list[float]:
    """Ask for and return the 5 personality trait values (normalized 0-1).

    >>> # Skipping doctest since it requires active stdin terminal user interaction.
    """
    print("\n--- Personality Survey (from 0 to 10) ---")

    def get_trait_input(prompt: str) -> float:
        """Prompt the user for a single trait and return its normalized value.

        >>> # Skipping doctest since it requires active stdin terminal user interaction.
        """
        val = -1.0
        while not 0 <= val <= 10:
            try:
                val = float(input(prompt))
                if not 0 <= val <= 10:
                    print("Error: Must enter a specific value from 0 to 10.")
            except ValueError:
                print("Error: Must enter a specific numerical value (0-10).")
        return val / 10

    o = get_trait_input("Your openness to new things: ")
    c = get_trait_input("Your level of organization and responsibility (Conscientiousness): ")
    e = get_trait_input("Your social extroversion (Extraversion): ")
    a = get_trait_input("Your affinity/teamwork ability (Agreeableness): ")
    s = get_trait_input("Your emotional expression/sensitivity (Expressiveness): ")
    return [o, c, e, a, s]


def get_love_lang() -> str:
    """Ask for and return the love language selection.

    >>> # Skipping doctest since it requires active stdin terminal user interaction.
    """
    print("\nWhat's your love language?")
    langs = ["Quality Time", "Physical Touch", "Acts of Service", "Receiving Gifts", "Words of Affirmation"]
    for idx, lang in enumerate(langs, 1):
        print(f"{idx}: {lang}")

    love_lang = ""
    while love_lang not in langs:
        try:
            lang_choice = int(input("Please select (1-5): "))
            if 1 <= lang_choice <= len(langs):
                love_lang = langs[lang_choice - 1]
            else:
                print("Error: Must enter a specific value from 1 to 5.")
        except ValueError:
            print("Error: Must enter a specific integer value.")
    return love_lang


def get_user_input() -> dict[str, Any]:
    """Return a dictionary of user answers based on terminal prompts.

    >>> # Skipping doctest since it requires active stdin terminal user interaction.
    """
    print("\n" + "=" * 50)
    print("--- CUPID MATCHING SYSTEM: PERSONAL PROFILE SURVEY ---")
    print("=" * 50)

    age, edu = get_demographics()
    career_field, ambition = get_career_info()
    traits = get_personality_traits()
    love_lang = get_love_lang()

    return {
        'age': age,
        'education': edu,
        'career_ambition': ambition,
        'openness': traits[0],
        'extraversion': traits[2],
        'agreeableness': traits[3],
        'conscientiousness': traits[1],
        'chronotype': 0.5,
        'spontaneity': 0.5,
        'emotional_expressiveness': traits[4],
        'location': 'Urban',
        'career_field': career_field,
        'love_language': love_lang
    }


def process_and_align_data(new_user_id: str, new_user_dict: dict[str, Any]) -> \
        (tuple[dict[str, list[float]], list[float]]):
    """Process dataset and new user together to align features and return vectors.

    Preconditions:
        - new_user_id != ''
        - new_user_dict != {}

    >>> test_dict = {'age': 25, 'education': 3, 'career_ambition': 0.8,
    ...              'openness': 0.5, 'extraversion': 0.5, 'agreeableness': 0.5,
    ...              'conscientiousness': 0.5, 'chronotype': 0.5, 'spontaneity': 0.5,
    ...              'emotional_expressiveness': 0.5, 'location': 'Urban',
    ...              'career_field': 'Tech', 'love_language': 'Quality Time'}
    >>> u_vecs, inter_vec = process_and_align_data("Test_User", test_dict)
    >>> 'Test_User' not in u_vecs
    True
    """
    users_data = data_loader.load_and_clean_data("cupid_algorithm_dataset.csv")
    combined_data = users_data + [{**new_user_dict, 'user_id': new_user_id}]

    # Encoding
    data = combined_data
    for col_name in data_loader.CATEGORICAL_COLUMNS:
        data = data_loader.apply_one_hot_encoding(data, col_name)

    # Normalization
    data = data_loader.normalize_numerical_data(data, data_loader.NUMERICAL_COLUMNS)

    # Feature column indexing
    feature_columns = data_loader.NUMERICAL_COLUMNS.copy()
    all_keys = list(data[0].keys())
    encoded_cols = sorted([key for key in all_keys if key not in feature_columns and key != 'user_id'])
    feature_columns.extend(encoded_cols)

    # Extraction
    user_vectors = {}
    new_user_vector = []
    for row in data:
        u_id = row['user_id']
        vector = [float(row[f_col]) for f_col in feature_columns]
        if u_id == new_user_id:
            new_user_vector = vector
        else:
            user_vectors[u_id] = vector
    return user_vectors, new_user_vector


def main() -> None:
    """Execute the main workflow of the Cupid Graphing recommendation system.

    >>> # Full workflow script execution cannot be doctested linearly due to terminal dependency.
    """
    print("Welcome to Cupid Interactive System!")

    print("\n[Step 1/4] Taking Personality Survey...")
    inter_dict = get_user_input()
    inter_id = "New_User_Interactive"

    print("\n[Step 2/4] Loading background datasets and establishing algorithms...")
    u_vecs, inter_vec = process_and_align_data(inter_id, inter_dict)

    # Step 3: Graph construction
    sample_dim = len(inter_vec)
    max_d = math.sqrt(sample_dim)
    sim_func = recommendation.SimilarityCalculator(max_d)

    print("[Step 3/4] Building Compatibility Graph...")
    graph = CompatibilityGraph(threshold=0.75)
    graph.add_users_as_nodes(u_vecs)
    graph.build_edges(sim_func)
    graph.add_new_user(inter_id, inter_vec, sim_func)

    # Step 4: BFS Logic
    print("[Step 4/4] Finding matches via transitive similarity...")
    matches = recommendation.weighted_bfs_recommendation(graph.graph, inter_id, top_n=5)

    print("\n" + "=" * 40 + "\n--- MATCHES FOUND ---" + "\n" + "=" * 40)
    for rank, (m_id, score) in enumerate(matches, 1):
        print(f"#{rank}: {m_id} (Compatibility Score: {score:.4f})")

    print("\nRendering visualization... check your browser.")
    visualizer.draw_interactive_cluster(graph.graph, inter_id, matches)


if __name__ == "__main__":
    main()
