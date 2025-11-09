"""
Script for choosing the founders of the simulation
"""

import random
import pandas as pd
import polars as pl
import numpy as np
import networkx as nx
import itertools as it
import re
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

def calculate_relatedness():
    pass

class Relatedness:
    pass

######## Methods for sampling   ########
######## unrelated individuals  ########

####### For simple unrelated sampling
def get_simple_graph(nodes: List[str], relatedness_dict: dict, max_k: float) -> nx.Graph:
    # Initialize the graph
    G = nx.Graph()
    G.add_nodes_from(nodes) # Add the nodes

    # Now susbset to only edges between sufficiently related individuals
    edge_list = [pair for pair, k in relatedness_dict.items() if k > max_k]
    G.add_edges_from(edge_list) # Add the edges between "related"

    return G

def simple_family_sample(G: nx.Graph, n_members: int):
    for _ in range(100): # Give it 100 attempts
        g = G.copy()
        nodes = g.nodes()
        out_fam = [] # Stores the IDs of the founders
        for _ in range(n_members):
            new_node = np.random.choice(list(nodes))
            out_fam.append(str(new_node))
            nodes = nodes - {new_node} - set(g.neighbors(new_node))
            if len(nodes) == 0:
                break
            g = g.subgraph(nodes)

        if len(out_fam) == n_members:
            return out_fam
        
    return set()


class SampleRelatives:

    def __init__(self, relatedness_dict: dict, sample_list: Union[list, np.ndarray] = None):

        self.kinship = relatedness_dict

        # Should supply a sample list! May miss out on unrelated indviduals
        if sample_list:
            self.samples = np.array(sample_list)
        else:
            samples = list(it.chain(*relatedness_dict.keys()))
            self.samples = np.array(set(samples))

        self.graph = nx.Graph()

    def set_mode(self, mode: str, **kwargs):

        self.mode = mode

        if mode == "simple":
            assert "max_k" in kwargs

            self.max_k = kwargs["max_k"]

            self.graph = get_simple_graph(self.samples,
                                          self.kinship,
                                          self.max_k)

    def get_unrelated_founders(self, n_members: int):

        if self.mode == "simple":
            fam = simple_family_sample(self.graph, n_members)

        # Ensure that a founder family was found
        assert len(fam) == n_members
        
        return fam




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


def update_fam_idx(sim_id, idx):
    return re.sub(r'(?<!\d)1(?!\d)', str(idx), sim_id, count=1)

def create_founders_file(
    vcf_samples: List[str],
    dry_run_families: List[List[str]],
    relatedness_dict: Dict[Tuple[str, str], float],
    n_sim: int,
    mode: str = "simple",
    max_k: float = None,
    **kwargs
) -> pd.DataFrame:
    """
    Generate a founders mapping file by assigning VCF samples to simulated pedigrees.

    Args:
        vcf_samples: list of sample IDs from the VCF
        dry_run_families: list of families (each family is a list of IDs, e.g. [["A","B","C"], ["D","E"]])
        relatedness_dict: dict {(id1, id2): relatedness coefficient}
        r: relatedness threshold for founders
        n_sim: number of simulated pedigrees per family

    Returns:
        pd.DataFrame with columns: dry_run_id, vcf_id
    """

    sampling = SampleRelatives(relatedness_dict, vcf_samples)

    # TODO implement other modes and ar
    if mode == "simple":
        assert max_k is not None
        sampling.set_mode("simple", max_k=max_k)

    assignments = []

    for fam in dry_run_families:
        for i in range(n_sim):
            founders = sampling.get_unrelated_founders(len(fam))
            for sim_id, sample_id in zip(fam, founders):
                assignments.append([sim_id, sample_id, i+1])

    founder_df = pd.DataFrame(assignments, columns=["sim_id", "sample_id", "fam_idx"])
    
    founder_df["sim_id"] = founder_df.apply(lambda x: update_fam_idx(x.sim_id, x.fam_idx), axis=1)

    return founder_df.drop("fam_idx", axis=1)
