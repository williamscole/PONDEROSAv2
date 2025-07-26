import pytest
import numpy as np
import polars as pl

from ponderosa.ibd_tools import ProcessSegments, Features


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

class TestProcessPairs:

    def test_basic(self):

        processed = ProcessSegments(BASIC_SEGMENTS, max_gap=1)

        chr1 = processed._get_ibd_stats([0, 1, 2, 3])

        assert chr1[Features.IBD1] == 45
        assert chr1[Features.IBD2] == 5
        assert chr1[Features.H1] == 30
        assert chr1[Features.H2] == 30
        assert chr1[Features.N] == 2

        df = pl.DataFrame(
                BASIC_SEGMENTS,
                schema=["chromosome", "start_cm", "end_cm", "id1_haplotype", "id2_haplotype"]
            )

        ibd_stats = processed.ibd_sharing_stats(df)

        for key, val in BASIC_SEGMENTS_TRUTH.items():
            assert ibd_stats[getattr(Features, key)] == val
