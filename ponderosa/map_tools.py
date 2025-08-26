import polars as pl
import pandas as pd
from pathlib import Path
from typing import Tuple, Union, List
import numpy as np

####### Load map file #######
class PlinkMap(pd.DataFrame):
    
    _metadata = ['map_len']
    
    def __init__(self, data: pd.DataFrame = None, *args, map_len: float = None, **kwargs):
        # Call parent DataFrame constructor with all args and kwargs
        if data is not None:
            super().__init__(data, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
        
        # Only calculate genome length if not provided and we have the required columns
        if map_len is not None:
            self.map_len = map_len
        elif self._has_required_columns():
            self.map_len = self._calculate_genome_length()
        else:
            self.map_len = 0.0
    
    def _has_required_columns(self) -> bool:
        """Check if DataFrame has the required columns for genome length calculation."""
        required_cols = ['chromosome', 'cm']
        return all(col in self.columns for col in required_cols)

    @classmethod
    def from_file(cls, plink_file: Union[str, Path]):
        """Load PlinkMap from a plink file."""
        data = pd.read_csv(
                        plink_file,
                        sep=r'\s+',  # Any whitespace
                        header=None,
                        names=["chromosome", "rsid", "cm", "bp"],
                        dtype={
                            "chromosome": 'int8',
                            "rsid": 'string',
                            "cm": 'float64',
                            "bp": 'int64'         
                        }
                    )
        return cls(data)

    def _calculate_genome_length(self) -> float:
        """Calculate total genome length from the map data."""
        if self.shape[0] == 0 or not self._has_required_columns():
            return 0.0
        
        tot = 0.0
        # Use regular pandas DataFrame operations to avoid recursion
        df = pd.DataFrame(self)  # Create a regular DataFrame copy
        for _, chrom_df in df.groupby("chromosome"):
            if len(chrom_df) > 1:
                tot += (chrom_df.iloc[-1]["cm"] - chrom_df.iloc[0]["cm"])

        return tot
    
    def get_col(self, col: str) -> np.ndarray:
        return self[col].values
    
    def filter_chrom(self, chrom: int) -> 'PlinkMap':
        filtered_df = self[self.chromosome == chrom]
        # Create new PlinkMap and recalculate map_len for the filtered chromosome
        new_obj = PlinkMap(filtered_df)
        return new_obj    
    
    @property
    def _constructor(self):
        """Ensures that operations return PlinkMap objects instead of DataFrames."""
        def constructor(*args, **kwargs):
            # For internal pandas operations, pass map_len=0.0 to avoid recalculation
            # Extract map_len from kwargs if present, otherwise default to 0.0
            map_len = kwargs.pop('map_len', 0.0)
            return PlinkMap(*args, map_len=map_len, **kwargs)
        return constructor
    
    @property
    def _constructor_sliced(self):
        """Constructor for Series operations."""
        return pd.Series
    
    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata during operations."""
        self = super().__finalize__(other, method, **kwargs)
        if hasattr(other, 'map_len'):
            self.map_len = other.map_len
        else:
            # For new objects created during operations, set map_len to 0
            # User can recalculate if needed
            self.map_len = 0.0
        return self
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, genome_length: float = None):
        """Create PlinkMap from existing DataFrame."""
        return cls(df, map_len=genome_length)
    
    def recalculate_map_length(self):
        """Explicitly recalculate the genome length."""
        self.map_len = self._calculate_genome_length()
        return self.map_len
    

class GeneticMap:

    def __init__(self, map_df: PlinkMap):

        self.map_df = map_df

        self.genome_len = map_df.map_len

    @classmethod
    def add_plink(cls, map_file: Union[str, Path]):

        map_df = PlinkMap.from_file(map_file)

        return cls(map_df)
    
    @classmethod
    def add_plink_list(cls, map_file_list: List[Union[str, Path]]):

        map_df_list = []
        for map_file in map_file_list:
            map_df = PlinkMap.from_file(map_file)
            map_df_list.append(map_df)

        map_df = PlinkMap.from_dataframe(pd.concat(map_df_list))

        return cls(map_df)

    @classmethod
    def add_hapmap(cls, hapmap_file: str):
        pass

    def _wget_hapmap(self, map_build: int) -> Tuple[pd.DataFrame, float]:
        map_url = f"https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh{map_build}.map.zip"
        pass

    @classmethod
    def no_map(cls, map_build: int = 37):
        map_df, genome_len = cls._wget_hapmap(map_build)
        return cls(map_df, genome_len)

    def interp(self, arr: np.ndarray, chrom_arr: np.ndarray) -> np.ndarray:

        interp_arr = np.zeros(arr.shape[0])

        for chrom in np.unique(chrom_arr):
            chrom_idx = np.where(chrom_arr == chrom)[0]

            chrom_map = self.map_df.filter_chrom(chrom)

            cm_val = chrom_map.get_col("cm")
            bp_val = chrom_map.get_col("bp")
                    
            interp_arr[chrom_idx] = np.interp(arr[chrom_idx], bp_val, cm_val)

        return interp_arr

    def get_genome_length(self):
        return self.genome_len
