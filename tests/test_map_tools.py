import pandas as pd
import pytest
import numpy as np
from pathlib import Path

test_dir = Path(__file__).parent

from ponderosa.map_tools import PlinkMap, GeneticMap

class TestPlinkMap:

    _map_list = [
        [1, "A", 0.0, 0],
        [1, "B", 5.0, 5],
        [1, "C", 10.0, 10],
        [2, "D", 0.0, 0],
        [2, "E", 5.0, 5],
        [2, "F", 15.0, 15]
    ]

    simple_map_df = pd.DataFrame(_map_list,
            columns=["chromosome", "rsid", "cm", "bp"])

    def test_basic(self):

        map_obj = PlinkMap(self.simple_map_df)

        chr1_obj = map_obj.filter_chrom(1)
        chr2_obj = map_obj.filter_chrom(2)

        assert map_obj.map_len == 25.0
        assert chr1_obj.map_len == 10.0
        assert chr2_obj.map_len == 15.0

    def test_single_file_test1(self):

        file_n = test_dir / "data" / "test1" / "test.map"

        map_obj = PlinkMap.from_file(file_n)

        assert map_obj.map_len == 320.0

        chr1_obj = map_obj.filter_chrom(1)
        chr2_obj = map_obj.filter_chrom(2)

        assert chr1_obj.map_len == 160.0
        assert chr2_obj.map_len == 160.0

    def test_multi_file_test1(self):

        file_n = test_dir / "data" / "test1" / "test_chr1.map"

        map_obj = PlinkMap.from_file(file_n)

        assert map_obj.map_len == 160.0


class TestGeneticMap:

    _map_list = [
    [1, "A", 0.0, 0], # chr 1 has 2x rr as chr 2
    [1, "B", 10.0, 5],
    [1, "C", 20.0, 10],
    [2, "D", 0.0, 0],
    [2, "E", 5.0, 5],
    [2, "F", 15.0, 15]
]

    simple_map_df = pd.DataFrame(_map_list,
            columns=["chromosome", "rsid", "cm", "bp"])

    def test_basic(self):

        map_df = PlinkMap(self.simple_map_df)

        gmap = GeneticMap(map_df)

        assert gmap.get_genome_length() == 35.0

        chr_arr = np.array([1, 1, 2, 2])
        bp_arr = np.array([2, 6, 3, 12])

        exp_cm = np.array([4.0, 12.0, 3.0, 12.0])
        obs_cm = gmap.interp(bp_arr, chr_arr)

        np.testing.assert_array_equal(exp_cm, obs_cm)

    def test_single_file_test1(self):

        file_n = test_dir / "data" / "test1" / "test.map"

        gmap = GeneticMap.add_plink(file_n)

        assert gmap.get_genome_length() == 320.0

    def test_multi_file_test1(self):

        file1 = test_dir / "data" / "test1" / "test_chr1.map"
        file2 = test_dir / "data" / "test1" / "test_chr2.map"

        gmap = GeneticMap.add_plink_list([file1, file2])

        assert gmap.get_genome_length() == 320.0






