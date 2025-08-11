"""
Script for choosing the founders of the simulation
"""

import random
import pandas as pd
import polars as pl
from ponderosa.data_loading import load_ibd_from_file

def choose_founders(
    ibd_file: str,
    ibd_caller: str,
    genetic_map_file: str,
    n_pairs: int,
    kinship_threshold: float = None,
    total_cm_threshold: float = None
) -> pd.DataFrame:
    """
    Randomly sample founder pairs from a pool of individuals in an IBD file.

    Requires:
        - Either `kinship_threshold` or `total_cm_threshold`
        - A genetic map file for IBD-to-cM conversion
    """

    if (kinship_threshold is None and total_cm_threshold is None) or \
       (kinship_threshold is not None and total_cm_threshold is not None):
        raise ValueError("You must provide exactly one of: kinship_threshold OR total_cm_threshold.")

    if kinship_threshold is not None:
        # TODO: Implement proper conversion using genetic_map_file
        total_cm_threshold = 1000  # <-- Replace this with actual conversion logic
        print(f"Converted kinship {kinship_threshold} to total_cm_threshold: {total_cm_threshold}")

    ibd_df_all = load_ibd_from_file(
        file_path = ibd_file,
        ibd_caller = ibd_caller,
        min_segment_length=0,
        min_total_ibd=0,
        #to_pandas=True
    )

    ibd_df_related = load_ibd_from_file(
        file_path = ibd_file,
        ibd_caller = ibd_caller,
        min_segment_length=None,  # TODO: Revisit this
        min_total_ibd=total_cm_threshold,
        #to_pandas=True
    )

    closely_related = set(
        tuple(sorted((row[0], row[1])))
        for row in ibd_df_related.select(["id1", "id2"]).iter_rows()
    )

    individuals = set(ibd_df_all["id1"].to_list()).union(set(ibd_df_all["id2"].to_list()))

    print("Closely related pairs:", closely_related)
    print("Number of individuals:", len(individuals))
    print("Sample individuals:", list(individuals)[:5])

    candidate_pairs = []
    individuals = list(individuals)
    for i in range(len(individuals)):
        for j in range(i + 1, len(individuals)):
            pair = tuple(sorted((individuals[i], individuals[j])))
            if pair not in closely_related:
                candidate_pairs.append(pair)

    if not candidate_pairs:
        raise ValueError("No unrelated pairs available under given threshold.")

    sampled = [random.choice(candidate_pairs) for _ in range(n_pairs)]

    return pl.DataFrame(sampled, schema=["id1", "id2"])
