import pytest
import numpy as np
import shutil
import os
import sys
import networkx as nx
from pathlib import Path
import itertools as it

from ponderosa.simulation.config import SimulationConfig, PedSimConfig, TrainingConfig
from ponderosa.simulation.setup import simulation_workspace, _symlink_files
from ponderosa.simulation.founders import calculate_relatedness, simple_family_sample, pedsim_dryrun, create_founders_file, relatedness_from_king, get_simple_graph
from ponderosa.simulation.simulate import _get_vcf_samples

@pytest.fixture(scope="session")
def scratch_dir():
    """Fixed scratch directory that's cleaned at the start of each test session"""
    scratch = Path("/oscar/scratch/cwilli50/simulations")
    
    # Clean everything in the directory at the start
    if scratch.exists():
        shutil.rmtree(scratch)
    
    scratch.mkdir(parents=True, exist_ok=True)
    
    yield scratch

def get_sim_file(sim_number: int, which_file: str):

    files = {
        "yaml": "args.yaml",
        "vcf": "founders.vcf",
        "vcf_gz": "founders.vcf.gz",
        "king": "king.seg"
    }

    return Path(f"tests/data/simulation{sim_number}/{files[which_file]}")


class TestConfig:

    def test_basic_1(self):

        yaml_file = get_sim_file(1, "yaml")

        config = SimulationConfig.from_yaml(yaml_file)


class TestSimulate:

    def test_get_vcf_samples_1(self):

        expected = np.array([f"ID{i}" for i in range(1, 11)])

        samples_vcf = np.array(_get_vcf_samples(get_sim_file(1, "vcf")))
        samples_vcf_gz = np.array(_get_vcf_samples(get_sim_file(1, "vcf_gz")))

        assert np.array_equal(samples_vcf, expected)
        assert np.array_equal(samples_vcf_gz, expected)


class TestSetup:

    def test_simulation_workspace_1(self):

        yaml_file = get_sim_file(1, "yaml")

        config = SimulationConfig.from_yaml(yaml_file)

        for cleanup in [False, True]:

            config.cleanup_temp = cleanup

            with simulation_workspace(config) as temp_dir:

                assert os.path.exists(temp_dir / "input.vcf.gz")
        
            # If cleanup == True, this the path should not exist (exist = False)
            assert os.path.exists(temp_dir / "input.vcf.gz") != cleanup

            if not cleanup:
                shutil.rmtree(temp_dir)
                

class TestFounders:

    def test_relatedness_1(self):

        king_file = get_sim_file(1, "king")

        r = relatedness_from_king(king_file)

    def test_get_simple_graph(self):
        relatives = ["A","B","C","D","E","F"]

        r_dict1 = {
            ("A","B"): 0.3,
            ("B","C"): 0.3,
            ("C","D"): 0.3,
            ("D","E"): 0.3,
            ("E","F"): 0.3
        }

        g = get_simple_graph(relatives, r_dict1, 0.5)
        assert len(list(nx.connected_components(g))) == len(relatives)

        g = get_simple_graph(relatives, r_dict1, 0.2)
        assert len(list(nx.connected_components(g))) == 1

        r_dict2 = {
            ("A","B"): 0.3,
            ("B","C"): 0.3,
            ("C","D"): 0.25,
            ("D","E"): 0.3,
            ("E","F"): 0.3
        }

        g = get_simple_graph(relatives, r_dict2, 0.28)
        assert len(list(nx.connected_components(g))) == 2

    def test_simple_family_sample(self):
        def _test(edges, n_members):
            g = nx.Graph()
            edges = [tuple(i) for i in edges]
            g.add_edges_from(edges)

            for _ in range(100):
                fam = tuple(simple_family_sample(g, n_members))

                for i in it.combinations(fam, r=2):
                    if tuple(i) in edges:
                        print(edges, n_members, fam)
                        assert False

            return fam


        edges1 = [
            ["A","B"],
            ["C","D"],
            ["D","E"]
        ]

        _test(edges1, 1) 
        _test(edges1, 2) 
        _test(edges1, 3) 

        edges2 = [
            ["A","B"],
            ["C","D"],
            ["D","E"],
            ["E","F"],
            ["F","A"]
        ]

        _test(edges2, 3)
        assert len(_test(edges2, 4)) == 0

       
