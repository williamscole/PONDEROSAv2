"""
Script for choosing the founders of the simulation
"""

import random
import pandas as pd
import polars as pl
import numpy as np
import itertools as it
from typing import Union, List, Tuple, Dict
from pathlib import Path
from ponderosa.data_loading import load_ibd_from_file, GeneticMap, FamFile
from ponderosa.simulation.pedsim import PedSim


########    Methods for computing kinship      ########
########    from KING files, IBD files, etc.   ########

def relatedness_from_ibd(
    ibd_file: List[str],
    ibd_caller: str,
    genetic_map_file_list: List[str],
) -> Dict[Tuple[str, str], float]:
    """
    Calculate and return relatedness coefficient among all pairs that share IBD.

    Output:
        dict with keys = (id1, id2), values = relatedness coefficient (0–1)
    """

    gm = GeneticMap.add_plink_list(genetic_map_file_list)
    total_genome_cm = gm.get_genome_length()
    print(f"Total genome length: {total_genome_cm:.2f} cm")

    ibd_df_related = load_ibd_from_file(
        file_paths = ibd_file,
        ibd_caller = ibd_caller,
        min_segment_length = 5,
        min_total_ibd = 5,
    )

    related_summary = (
        ibd_df_related
        .group_by(["id1", "id2"])
        .agg(pl.col("length_cm").sum().alias("total_ibd_cm"))
        .with_columns(
            (pl.col("total_ibd_cm") / total_genome_cm).alias("relatedness_coefficient")
        )
    )

    # Factor of 2 is to get it in the same scale as KING
    relatedness_dict = {
        (row["id1"], row["id2"]): row["relatedness_coefficient"] / 2
        for row in related_summary.to_dicts()
    }

    return relatedness_dict

def relatedness_from_king(king_file: Union[str, Path]) -> Dict[Tuple[str, str], float]:

    king_df = pd.read_csv(king_file, sep="\\s+")

    return {(row.ID1, row.ID2): row.PropIBD for row in king_df.itertuples()}

######## Methods for sampling   ########
######## unrelated individuals  ########
class SampleRelatives:

    def __init__(self, relatedness_dict: dict, sample_list: Union[list, np.ndarray] = None):

        self.kinship = relatedness_dict

        # Should supply a sample list! May miss out on unrelated indviduals
        if sample_list:
            self.samples = np.array(sample_list)
        else:
            samples = list(it.chain(*relatedness_dict.keys()))
            self.samples = np.array(set(samples))

    def set_mode(self, mode: str, **kwargs):

        self.mode = mode

        if mode == "simple":
            assert "max_k" in kwargs

            # TODO: create kinship graph here!

            self.max_k = kwargs["max_k"]

    def get_unrelated_family(self, n_members: int):

        assert self.mode == "simple"







#should return nested families from the dryrun [[“A”,”B”,”C”],[“D”,”E”,”F”]]
def pedsim_dryrun(
    pedsim: PedSim
) -> List[List[str]]:
    """   
    Ped-sim dry run to obtain fam file needed for simulations
    """

    pedsim.dry_run()
    fam_file = pedsim.get_file("fam")
    fam = FamFile(fam_file)
    return fam.get_founders()


def create_founders_file(
    vcf_samples: List[str],
    dry_run_families: List[List[str]],
    relatedness_dict: Dict[Tuple[str, str], float],
    r: float,
    n: int
) -> pd.DataFrame:
    """
    Generate a founders mapping file by assigning VCF samples to simulated pedigrees.

    Args:
        vcf_samples: list of sample IDs from the VCF
        dry_run_families: list of families (each family is a list of IDs, e.g. [["A","B","C"], ["D","E"]])
        relatedness_dict: dict {(id1, id2): relatedness coefficient}
        r: relatedness threshold for founders
        n: number of simulated pedigrees per family

    Returns:
        pd.DataFrame with columns: dry_run_id, vcf_id
    """

    def are_unrelated(selected: List[str], candidate: str) -> bool:
        """Check candidate against all already selected samples for relatedness ≤ r."""
        for s in selected:
            # lookup both (s, candidate) and (candidate, s) since dict might not be symmetric
            rel = relatedness_dict.get((s, candidate)) or relatedness_dict.get((candidate, s)) or 0.0
            if rel > r:
                return False
        return True

    assignments = []

    for fam in dry_run_families:
        for _ in range(n):
            chosen = []
            for person in fam:
                # Try to pick a sample that is unrelated to already chosen samples
                possible = [s for s in vcf_samples if s not in chosen and are_unrelated(chosen, s)]
                if not possible:
                    raise ValueError(f"Could not find unrelated sample for {person} in family {fam}")
                selected = random.choice(possible)
                chosen.append(selected)
                assignments.append({"dry_run_id": person, "vcf_id": selected})

    return pd.DataFrame(assignments)
