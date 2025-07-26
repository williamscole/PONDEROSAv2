import numpy as np
from typing import Tuple, List, Dict

# CONSTANTS
PATERNAL = 1
MATERNAL = 2

def format_pedigree_paths(paths) -> np.ndarray:

    if isinstance(paths, np.ndarray):
        paths = paths.tolist()

    elif isinstance(paths, list):    
        longest_path = max([len(path) for path in paths])
        
        # Add padding to the path if necessary (supports inter-generational)
        paths = [path + [0]*(longest_path-len(path)) for path in paths]

    return np.array(sorted(paths))

def get_parental_paths(path_dict: dict) -> Tuple[list, list]:

    p1 = sorted([[PATERNAL] + path for path in path_dict.get(PATERNAL, [])], key=lambda x: x[1:])
    p2 = sorted([[MATERNAL] + path for path in path_dict.get(MATERNAL, [])], key=lambda x: x[1:])

    return p1, p2