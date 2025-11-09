import pytest
import numpy as np
import pandas as pd
import shutil
import os
import sys
import re
import networkx as nx
from pathlib import Path
import itertools as it

from ponderosa.simulation.config import SimulationConfig, PedSimConfig, TrainingConfig
from ponderosa.simulation.setup import simulation_workspace, _symlink_files
from ponderosa.simulation.founders import calculate_relatedness, simple_family_sample, pedsim_dryrun, create_founders_file, relatedness_from_king, get_simple_graph, update_fam_idx
from ponderosa.simulation.simulate import _get_vcf_samples
from ponderosa.simulation.pedsim import PedSim

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
        "king": "king.seg",
        "def": "second.def"
    }

    return Path(f"tests/data/simulation{sim_number}/{files[which_file]}")


class TestConfig:

    def test_basic_1(self):

        yaml_file = get_sim_file(1, "yaml")

        config = SimulationConfig.from_yaml(yaml_file)


class TestSetup:
    def test_simulation_workspace(self):
        """Test workspace creation for simulations 1 and 2."""
        
        for sim in [1, 2]:
            yaml_file = get_sim_file(sim, "yaml")
            config = SimulationConfig.from_yaml(yaml_file)
            config.cleanup_temp = True
            
            config.cleanup_temp = False
            
            with simulation_workspace(config) as temp_dir:
                print(f"\n{'='*60}")
                print(f"SIMULATION {sim} WORKSPACE")
                print(f"{'='*60}")
                print(f"Location: {temp_dir}")
                print(f"\nContents:")
                
                # List files with sizes
                for item in sorted(temp_dir.iterdir()):
                    if item.is_symlink():
                        target = item.resolve()
                        # Get size of target file
                        if target.exists():
                            size_bytes = target.stat().st_size
                            size_mb = size_bytes / (1024 * 1024)
                            print(f"  📎 {item.name} -> {target} ({size_bytes:,} bytes / {size_mb:.2f} MB)")
                        else:
                            print(f"  📎 {item.name} -> {target} (TARGET MISSING!)")
                    else:
                        size_bytes = item.stat().st_size
                        size_mb = size_bytes / (1024 * 1024)
                        print(f"  📄 {item.name} ({size_bytes:,} bytes / {size_mb:.2f} MB)")
                
                # Verify VCF exists
                assert (temp_dir / "input.vcf.gz").exists(), \
                    f"VCF file not found in {temp_dir}"
                
                print(f"{'='*60}\n")
            
            # Clean up
            shutil.rmtree(temp_dir)
            print(f"✓ Cleaned up temp directory: {temp_dir}\n")

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

    def test_update_fam_idx(self):
        iid = "avuncular1_g3-b1-i1"
        i = 10
        target = "avuncular10_g3-b1-i1"
        assert update_fam_idx(iid, i) == target

    def test_create_founders_file(self):

        vcf_samples = ["A", "B", "C", "D", "E", "F"]
        dry_run_families = [["A1_1","A1_2", "A1_3"],
                            ["B1_1", "B1_2"]]
        relatedness_dict = {("A","B"): 0.5}
        max_k = 0.2
        n_sim = 3

        df = create_founders_file(vcf_samples,
                             dry_run_families,
                             relatedness_dict,
                             n_sim,
                             mode="simple",
                             max_k = max_k)

        df["fam"] = df.sim_id.apply(lambda x: x[0])
        df["fam_idx"] = df.sim_id.apply(lambda x: x[1])

        for (fam, idx), tmp in df.groupby(["fam", "fam_idx"]):
            pairs = list(it.combinations(tmp.sample_id.values, r=2))
            assert ("A","B") not in pairs
            assert ("B","A") not in pairs

        for i in it.chain(*dry_run_families):
            for k in range(1, n_sim+1):
                i = i.replace("1_", f"{k}_")
                assert i in df.sim_id.values

class TestSimulate:

    def _get_fam(self, sample_id):
        return re.split(r'\d+', sample_id)[0]

    def _get_fam_idx(self, sample_id):
        match = re.search(r'\d+', sample_id)
        if match:
            return match.group()
        return None

    def test_basic(self):

        yaml_file = get_sim_file(2, "yaml")

        config = SimulationConfig.from_yaml(yaml_file)
        config.cleanup_temp = True

        vcf_ext = ".vcf.gz" if str(config.pedsim.vcf_file).endswith('.gz') else ".vcf"

        with simulation_workspace(config) as temp_dir:

            # Init the pedsim object
            pedsim = PedSim(executable_path=str(config.pedsim.pedsim_executable),
                            output=temp_dir / "simulation",
                            vcf_ext=vcf_ext)

            vcf_samples = _get_vcf_samples(vcf_file=Path(pedsim.get_input("vcf")))

            relatives = []
            relatedness_dict = {}
            for i, j in it.combinations(vcf_samples, r=2):
                if np.random.choice([0,1], p=[0.975, 0.025]):
                    relatives += [(i,j), (j,i)]
                    relatedness_dict[(i,j)] = 0.5

            dry_run_families = pedsim_dryrun(pedsim)

            vcf_samples = _get_vcf_samples(vcf_file=Path(pedsim.get_input("vcf")))

            founders_df = create_founders_file(
            vcf_samples=vcf_samples,
            dry_run_families=dry_run_families,
            relatedness_dict=relatedness_dict,
            max_k=0.25,
            n_sim=10
        )

            founders_df["fam"] = founders_df.sim_id.apply(self._get_fam)
            founders_df["fam_idx"] = founders_df.sim_id.apply(self._get_fam_idx)

            for _, tmp in founders_df.groupby(["fam", "fam_idx"]):
                for pair in it.combinations(tmp.sample_id.values, r=2):
                    assert pair not in relatives

            founders_df = founders_df.drop(labels=["fam", "fam_idx"], axis=1)

            founders_file = temp_dir / "founders.txt"
            founders_df.to_csv(founders_file, sep="\t", index=False, header=False)
            
            # 7. Update PedSim with founders and execute
            pedsim.update_flag("--set_founders", str(founders_file))
            pedsim.execute()

            # print(pedsim.get_file("fam"))

            # fam_df = pd.read_csv(pedsim.get_file("fam"), delim_whitespace=True, header=None)

            pytest.set_trace()


# TODO: Nov 9 — implement debugging flags in pedsim to inspect output. Currently, the sim is running and being created with is good!!!!



            


       
