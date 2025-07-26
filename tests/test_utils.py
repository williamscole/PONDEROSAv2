import numpy as np
import pytest

from ponderosa.utils import remove_nan

class TestRemoveNaN:

    def _generate_matrix_with_nans(self, n: int, nan_row_indices: list, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        
        matrix = np.random.rand(n, 4)
        
        for row_idx in nan_row_indices:
            num_nans = np.random.randint(1, 5)
            nan_col_indices = np.random.choice(4, size=num_nans, replace=False)
            matrix[row_idx, nan_col_indices] = np.nan
        
        return matrix

    def _generate_pair_list(self, n_rows):
        i = 0
        pair_list = []
        for _ in range(n_rows):
            pair_list.append([str(i), str(i+1)])
            i += 2

        return np.array(pair_list)

    @pytest.mark.parametrize("n_rows,nan_row_indices", [
            pytest.param(5, [3], id="basic1"),
            pytest.param(5, [0,1,3,4], id="basic2"), 
            pytest.param(5, [0,1,2,3,4], id="basic3"), 
            ])
    # TODO deal with zero entries?
    def test_basic(self, n_rows, nan_row_indices):

        X = self._generate_matrix_with_nans(n_rows, nan_row_indices)

        pair_list = self._generate_pair_list(n_rows)

        masked_pairs, masked_X = remove_nan(pair_list, X)

        assert masked_pairs.shape[0] == n_rows - len(nan_row_indices)
        assert masked_X.shape[0] == n_rows - len(nan_row_indices)

        for index in nan_row_indices:
            id1, id2 = pair_list[index]
            assert id1 not in masked_pairs
            assert id2 not in masked_pairs

