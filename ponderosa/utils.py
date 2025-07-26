import numpy as np
from typing import Tuple, List, Set, Dict, Any

def remove_nan(pair_list: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Only rows that contain no NaN
    valid_mask = ~np.isnan(X).any(axis=1)

    return pair_list[valid_mask], X[valid_mask]


