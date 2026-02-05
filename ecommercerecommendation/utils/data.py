"""
This script has utility functions to handle data.

"""

import os
import re
from subprocess import call
from typing import Any, List, Set

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

BULLET_POINT = "\n- "
REMOVE_CHARS = ".,'\"&/- !()"


def get_data(name: str = "data") -> pd.DataFrame:
    """
    Loads the initial dataset in case it is not loaded and saved in the cache
    folder. Returns the corresponding data from the cache folder or an empty
    DataFrame if the CSV is not available.

    :param name: str, name of the CSV file, without the extension
    :return: the pandas DataFrame of the corresponding dataset
    """
    local_data_path = f"cache/{name}.csv"
    if not os.path.isfile(local_data_path):
        # let us skip dealing with exceptions and errors in this case
        if name == "data":
            call("./download_dataset.sh")
        else:
            return pd.DataFrame()

    return pd.read_csv(
        local_data_path,
        parse_dates=["InvoiceDate"] if name != "data_rule_based" else None,
        encoding="unicode_escape" if name == "data" else "utf-8",
    )


def venn_sets(set1: Set[Any], set2: Set[Any]) -> str:
    """
    To describe what a Venn diagram would show in numbers: cardinality of
    union, intersection and the differences.

    :param set1: set of any elements
    :param set2: set of any elements
    :return: the above describe numbers in text
    """

    cardinality_union = len(set1 | set2)
    cardinality_intersection = len(set1 & set2)
    cardinality_set1_2 = len(set1 - set2)
    cardinality_set2_1 = len(set2 - set1)

    return (
        f"[{cardinality_union}]: "
        f"[ {cardinality_set1_2} - "
        f"({cardinality_intersection}) - "
        f"{cardinality_set2_1} ]"
    )


def bullet_points(list_str: List[str]) -> str:
    """
    Creates a string with bullet points out of a list of strings.

    :param list_str: separate bullet points
    :return: as a string, concatenated
    """
    return BULLET_POINT + BULLET_POINT.join(list_str)


def remove_chars(s: str) -> str:
    """
    Removes all non-alphanumeric characters, that we want to check about,
    from the string.

    :param s: description
    :return: text without those characters
    """
    s_temp = s
    for c in REMOVE_CHARS:
        s_temp = s_temp.replace(c, "")
    return s_temp


def clean_entry(text: str) -> str:
    """
    Text cleaning functions as presented in the 2nd notebook.

    :param text: description
    :return: the clean description
    """

    return " ".join(
        re.sub(
            r"([0123456789].\")",
            r"\1 ",
            re.sub(
                r"[!()/,]",
                " ",
                text.strip(".")
                .replace(". ", " ")
                .replace("&", " AND ")
                .replace("/", " ")
                .replace(" - ", " "),
            ),
        )
        .strip(" ")
        .split()
    )


def find_minimax_assignment(dist_matrix: np.array):
    """
    Finds the unique assignment of antecedents to selected products
    that minimizes the maximum distance.
    """
    # Rows = Antecedents, Cols = Selected Products
    N, K = dist_matrix.shape

    # We can only match as many items as the smaller dimension
    max_possible_matches = min(N, K)

    # 1. Get all unique distance values in the matrix and sort them
    unique_dists = np.unique(dist_matrix)

    low = 0
    high = len(unique_dists) - 1
    best_threshold = unique_dists[high]

    # 2. Binary Search for the optimal threshold (The Bottleneck)
    while low <= high:
        mid = (low + high) // 2
        threshold = unique_dists[mid]

        # Create a bipartite graph: edge exists if distance <= threshold
        adj_matrix = (dist_matrix <= threshold).astype(int)
        sparse_graph = csr_matrix(adj_matrix)

        # Find maximum bipartite matching size
        matching = maximum_bipartite_matching(sparse_graph, perm_type="column")
        actual_matches = np.count_nonzero(matching != -1)

        if actual_matches >= max_possible_matches:
            best_threshold = threshold
            high = mid - 1  # Try a tighter (smaller) bottleneck
        else:
            low = mid + 1  # Need a looser (larger) bottleneck

    return best_threshold
