"""This module reads the dataset, extracts unique users from the
a_* and b_* columns, applies one-hot encoding to categorical variables,
normalizes numerical variables, and produces feature vectors for each user.
"""

from __future__ import annotations
import csv
from typing import Any

NUMERICAL_COLUMNS = [
    'age',
    'education',
    'career_ambition',
    'openness',
    'extraversion',
    'agreeableness',
    'conscientiousness',
    'chronotype',
    'spontaneity',
    'emotional_expressiveness'
]

CATEGORICAL_COLUMNS = [
    'location',
    'career_field',
    'love_language'
]


def load_and_clean_data(filepath: str, max_users: int = 4000) -> list[dict[str, Any]]:
    """Return a list of user data dictionaries from the given CSV file,
    with duplicate users removed, capped at max_users to prevent performance explosion.

    The original CSV stores paired users in columns starting with a_ and b_.
    This function separates them into one unified user table and removes duplicates.

    Preconditions:
        - filepath != ''

    >>> import tempfile, os
    >>> temp_csv = tempfile.NamedTemporaryFile(mode='w+', delete=False, newline='', encoding='utf-8-sig')
    >>> header = ['a_age','a_education','a_location','a_career_field','a_career_ambition','a_openness','a_extraversion','a_agreeableness','a_conscientiousness','a_chronotype','a_spontaneity','a_love_language','a_emotional_expressiveness', 'b_age','b_education','b_location','b_career_field','b_career_ambition','b_openness','b_extraversion','b_agreeableness','b_conscientiousness','b_chronotype','b_spontaneity','b_love_language','b_emotional_expressiveness']
    >>> _ = temp_csv.write(','.join(header) + '\\n')
    >>> row1 = ['20','1','NY','IT','0.5','0.5','0.5','0.5','0.5','0.5','0.5','Time','0.5'] * 2
    >>> _ = temp_csv.write(','.join(row1) + '\\n')
    >>> temp_csv.close()
    >>> data = load_and_clean_data(temp_csv.name, max_users=10)
    >>> len(data) == 1
    True
    >>> os.remove(temp_csv.name)
    """
    users = []
    visited = set()
    with open(filepath, 'r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            user_a = _extract_user(row, 'a_')
            user_b = _extract_user(row, 'b_')
            signature_a = _row_signature(user_a)
            signature_b = _row_signature(user_b)
            if signature_a not in visited:
                visited.add(signature_a)
                users.append(user_a)
                if len(users) >= max_users:
                    break
            if signature_b not in visited:
                visited.add(signature_b)
                users.append(user_b)
                if len(users) >= max_users:
                    break
    for i in range(len(users)):
        users[i]['user_id'] = f'User_{i}'
    return users


def apply_one_hot_encoding(data: list[dict[str, Any]], column_name: str) -> list[dict[str, Any]]:
    """Return the data list with the specified categorical column converted into
    one-hot encoded features.

    This ensures all categories have an equal mathematical distance.

    Preconditions:
        - column_name != ''

    >>> rows = [{'user_id': 'U1', 'love_language': 'Quality Time'},
    ...         {'user_id': 'U2', 'love_language': 'Physical Touch'}]
    >>> encoded = apply_one_hot_encoding(rows, 'love_language')
    >>> encoded[0]['love_language_Quality_Time']
    1
    >>> encoded[0]['love_language_Physical_Touch']
    0
    >>> 'love_language' in encoded[0]
    False
    """
    if not data:
        return []
    if column_name not in data[0]:
        return [entry.copy() for entry in data]
    categories = sorted({entry[column_name] for entry in data})
    encoded_data = []
    for row in data:
        new_row = row.copy()
        value = new_row.pop(column_name)
        for category in categories:
            safe_name = category.replace(' ', '_')
            new_key = f'{column_name}_{safe_name}'
            if value == category:
                new_row[new_key] = 1
            else:
                new_row[new_key] = 0
        encoded_data.append(new_row)
    return encoded_data


def normalize_numerical_data(
        data: list[dict[str, Any]],
        numerical_columns: list[str]) -> list[dict[str, Any]]:
    """Return the data list with numerical values scaled to a range between 0 and 1.

    Preconditions:
        - data != []
        - numerical_columns != []

    >>> rows = [{'x': 10, 'y': 100}, {'x': 20, 'y': 300}]
    >>> normed = normalize_numerical_data(rows, ['x', 'y'])
    >>> normed[0]['x']
    0.0
    >>> normed[1]['x']
    1.0
    """
    if not data:
        return []
    mins = {}
    maxes = {}
    for column in numerical_columns:
        values = []
        for row in data:
            values.append(float(row[column]))
        mins[column] = min(values)
        maxes[column] = max(values)
    normalized = []
    for row in data:
        new_row = row.copy()
        for column in numerical_columns:
            min_value = mins[column]
            max_value = maxes[column]
            value = float(row[column])
            if max_value == min_value:
                new_row[column] = 0.0
            else:
                new_row[column] = (value - min_value) / (max_value - min_value)
        normalized.append(new_row)
    return normalized


def generate_user_vectors(filepath: str, max_users: int = 4000) -> dict[str, list[float]]:
    """Return a dictionary mapping user IDs to their final numerical feature vectors.
    This function combines all cleaning and formatting steps.

    Preconditions:
        - filepath != ''
        - max_users > 0

    >>> import tempfile, os
    >>> temp_csv = tempfile.NamedTemporaryFile(mode='w+', delete=False, newline='', encoding='utf-8-sig')
    >>> header = ['a_age','a_education','a_location','a_career_field','a_career_ambition','a_openness','a_extraversion','a_agreeableness','a_conscientiousness','a_chronotype','a_spontaneity','a_love_language','a_emotional_expressiveness', 'b_age','b_education','b_location','b_career_field','b_career_ambition','b_openness','b_extraversion','b_agreeableness','b_conscientiousness','b_chronotype','b_spontaneity','b_love_language','b_emotional_expressiveness']
    >>> _ = temp_csv.write(','.join(header) + '\\n')
    >>> row1 = ['20','1','NY','IT','0.5','0.5','0.5','0.5','0.5','0.5','0.5','Time','0.5'] * 2
    >>> _ = temp_csv.write(','.join(row1) + '\\n')
    >>> temp_csv.close()
    >>> vecs = generate_user_vectors(temp_csv.name, max_users=10)
    >>> len(vecs['User_0']) > 5
    True
    >>> os.remove(temp_csv.name)
    """
    data = load_and_clean_data(filepath, max_users)
    if not data:
        return {}
    for col in CATEGORICAL_COLUMNS:
        data = apply_one_hot_encoding(data, col)
    data = normalize_numerical_data(data, NUMERICAL_COLUMNS)
    feature_columns = NUMERICAL_COLUMNS.copy()
    encoded_columns = []
    for key in data[0]:
        if key not in {'user_id'} and key not in NUMERICAL_COLUMNS:
            encoded_columns.append(key)
    encoded_columns.sort()
    feature_columns.extend(encoded_columns)
    user_vectors = {}
    for row in data:
        user_id = row['user_id']
        vector = []
        for column in feature_columns:
            vector.append(float(row[column]))
        user_vectors[user_id] = vector
    return user_vectors


def _extract_user(row: dict[str, str], prefix: str) -> dict[str, Any]:
    """Extract a single user dictionary from a row using the given prefix.

    Preconditions:
        - prefix in {'a_', 'b_'}
        - f'{prefix}age' in row

    >>> row_data = {'a_age': '25', 'a_education': '3', 'a_location': ' NY ', 'a_career_field': ' Tech ', 'a_career_ambition': '0.8', 'a_openness': '0.9', 'a_extraversion': '0.5', 'a_agreeableness': '0.7', 'a_conscientiousness': '0.8', 'a_chronotype': '0.4', 'a_spontaneity': '0.6', 'a_love_language': ' Gifts ', 'a_emotional_expressiveness': '0.9'}
    >>> user = _extract_user(row_data, 'a_')
    >>> user['age'] == 25 and user['location'] == 'NY'
    True
    """
    return {
        'age': int(row[f'{prefix}age']),
        'education': int(row[f'{prefix}education']),
        'location': row[f'{prefix}location'].strip(),
        'career_field': row[f'{prefix}career_field'].strip(),
        'career_ambition': float(row[f'{prefix}career_ambition']),
        'openness': float(row[f'{prefix}openness']),
        'extraversion': float(row[f'{prefix}extraversion']),
        'agreeableness': float(row[f'{prefix}agreeableness']),
        'conscientiousness': float(row[f'{prefix}conscientiousness']),
        'chronotype': float(row[f'{prefix}chronotype']),
        'spontaneity': float(row[f'{prefix}spontaneity']),
        'love_language': row[f'{prefix}love_language'].strip(),
        'emotional_expressiveness': float(row[f'{prefix}emotional_expressiveness'])
    }


def _row_signature(row: dict[str, Any]) -> tuple:
    """Return a hashable signature for duplicate detection.

    Preconditions:
        - row != {}

    >>> _row_signature({'b': 2, 'a': 1})
    (('a', 1), ('b', 2))
    """
    return tuple((key, row[key]) for key in sorted(row))


if __name__ == '__main__':
    import doctest

    doctest.testmod()

    # Uncomment when ready
    # import python_ta
    # 
    # python_ta.check_all(config={
    #     'extra-imports': ['csv'],
    #     'allowed-io': ['load_and_clean_data'],
    #     'max-line-length': 120
    # })
