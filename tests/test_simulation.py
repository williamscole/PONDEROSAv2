import pytest
import numpy as np
import shutil
import os
import sys
from pathlib import Path

from ponderosa.simulation.config import SimulationConfig, PedSimConfig, TrainingConfig
from ponderosa.simulation.setup import simulation_workspace, _symlink_files
from ponderosa.simulation.founders import calculate_relatedness, pedsim_dryrun, create_founders_file, Relatedness
from ponderosa.simulation.simulate import _get_vcf_samples

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

        r = Relatedness.from_king(king_file)
