import sys
from pathlib import Path
import polars as pl
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ponderosa.data_loading import PhasedIBDLoader, Individuals, Pairs, load_individuals, load_pairs
from ponderosa.ibd_tools import IBD, ProcessSegments, Features

# from ponderosa.ibd_tools import Pairs, ProcessSegments

BASIC_SEGMENTS = np.array([
            [1, 10, 20, 0, 0],
            [1, 20, 30, 0, 0],
            [1, 50, 70, 1, 1],
            [1, 65, 80, 0, 0],
            [2, 0, 50, 1, 1]
        ])

BASIC_SEGMENTS_TRUTH = {
    "H1": 80,
    "H2": 80,
    "N": 3,
    "N2": 1,
    "IBD1": 95,
    "IBD2": 5
}


class TestPairs:

    def test_basic(self):

        segments = np.array([
                [1, 10, 20, 0, 0],
                [1, 20, 30, 0, 0],
                [1, 50, 70, 1, 1],
                [1, 65, 80, 0, 0],
                [2, 0, 50, 1, 1]
            ])

        df = pl.DataFrame(
                    segments,
                    schema=["chromosome", "start_cm", "end_cm", "id1_haplotype", "id2_haplotype"]
                )

        pair_list = np.array([("A", "B"), ("B", "C"), ("C", "D")])

        dfs = []
        for id1, id2 in pair_list:
            tmp = df.clone()
            tmp = tmp.with_columns(pl.lit(id1).alias("id1"))
            tmp = tmp.with_columns(pl.lit(id2).alias("id2"))
            dfs.append(tmp)

        df = pl.concat(dfs)

        pairs = Pairs.from_segment_df(df, max_gap = 1)

        flattened = pairs.get_pair_data_from(pair_list, "IBD1", "IBD2", output_style="flatten")

        test_mat = np.array([[BASIC_SEGMENTS_TRUTH["IBD1"], BASIC_SEGMENTS_TRUTH["IBD2"]] for _ in pair_list])

        np.testing.assert_array_equal(flattened, test_mat)

    def test_out_of_order(self):

        pair_list1 = np.array([("A", "B"), ("B", "C"), ("C", "D")])
        pair_list2 = np.array([("B", "A"), ("B", "C"), ("D", "C")])

        pair_data = {
            ("A", "B"): {"A": -1, "B": 1},
            ("B", "A"): {"A": -1, "B": 1},
            ("B", "C"): {"B": -2, "C": 2},
            ("C", "B"): {"B": -2, "C": 2},
            ("C", "D"): {"C": -3, "D": 3},
            ("D", "C"): {"C": -3, "D": 3}
        }

        data1 = np.zeros((3, Features.NO_FEATURES))
        data2 = np.zeros((3, Features.NO_FEATURES))

        for data, pair_list in zip([data1, data2], [pair_list1, pair_list2]):
            for index, (id1, id2) in enumerate(pair_list):
                data[index,[Features.H1, Features.H2]] += np.array([pair_data[(id1, id2)][id1], pair_data[(id1, id2)][id2]])
                data[index,[Features.H1_ERR, Features.H2_ERR]] += np.array([pair_data[(id1, id2)][id1], pair_data[(id1, id2)][id2]])


        df1 = pl.DataFrame(
                    data1,
                    schema=Features.get_feature_names()
                )

        df2 = pl.DataFrame(
                    data2,
                    schema=Features.get_feature_names()
                )


        df1 = df1.with_columns(pl.Series("id1", pair_list1[:,0]))
        df1 = df1.with_columns(pl.Series("id2", pair_list1[:,1]))

        df2 = df2.with_columns(pl.Series("id1", pair_list2[:,0]))
        df2 = df2.with_columns(pl.Series("id2", pair_list2[:,1]))

        pairs1 = Pairs(df1)
        pairs2 = Pairs(df2)

        np.random.shuffle(pair_list1)
        np.random.shuffle(pair_list2)

        for pair_list in [pair_list1, pair_list2]:
            for pairs in [pairs1, pairs2]:
                pair_mat = pairs.get_pair_data_from(pair_list, "H1", "H2", output_style="flatten")
                for index, (id1, id2) in enumerate(pair_list):
                    h1, h2 = pair_mat[index,:]
                    assert pair_data[(id1, id2)][id1] == h1
                    assert pair_data[(id1, id2)][id2] == h2

        
        pair_mat = pairs.get_pair_data_from(np.array([("A", "B"),("Y", "Z")]), "H1", "H2", output_style="flatten")

        is_nan = np.isnan(pair_mat)
        assert is_nan[0,0] == False
        assert is_nan[0,1] == False
        assert is_nan[1,0] == True
        assert is_nan[1,1] == True

# def test_groupby(df: pl.DataFrame) -> pl.DataFrame:
#     return df.head(1)

# test_df = pl.DataFrame({
# "id1": ["A", "A", "B", "B", "A"],
# "id2": ["X", "Y", "X", "Y", "X"], 
# "value": [1, 2, 3, 4, 10]
# })


# if __name__ == "__main__":

#     if sys.argv[1] == "test1":

#         from test_config import config1

#         loader = PhasedIBDLoader(min_len=3.0, min_total_ibd=100.0)

#         segments = loader.load_filtered_segments(config1.files.ibd)

#         individuals = load_individual_files(config1.files)

#         pairs = Pairs(segments)

#         data = individuals.retrieve_data("MOTHER", "FATHER", output_style="zip")

#         import ipdb; ipdb.set_trace()

