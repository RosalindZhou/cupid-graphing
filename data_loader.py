# Reading the CSV file, cleaning the data, and applying one-hot encoding.

from __future__ import annotations
import csv
from typing import Any

# implement the functions, doctests


def load_and_clean_data(filepath: str) -> list[dict[str, Any]]:
    """Return a list of user data dictionaries from the given CSV file,
    with duplicate users removed.

    Preconditions:
    - filepath is the filename of a valid CSV data file
    """


def apply_one_hot_encoding(data: list[dict[str, Any]], column_name: str) -> list[dict[str, Any]]:
    """Return the data list with the specified categorical column converted into
    one-hot encoded features.

    This ensures all categories have an equal mathematical distance.
    """


def normalize_numerical_data(data: list[dict[str, Any]], numerical_columns: list[str]) -> list[dict[str, Any]]:
    """Return the data list with numerical values scaled to a range between 0 and 1.
    """


def generate_user_vectors(filepath: str) -> dict[str, list[float]]:
    """Return a dictionary mapping user IDs to their final numerical feature vectors.
    This function combines all cleaning and formatting steps.
    """
