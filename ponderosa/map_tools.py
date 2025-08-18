import polars as pl
import pandas as pd
from pathlib import Path
from typing import Tuple, Union
import numpy as np

####### Load map file #######
class PlinkMap(pl.DataFrame):
    
    def __init__(self, plink_file: Union[str, Path]):
        
        data = pl.read_csv(
                        plink_file,
                        separator=" ",
                        has_header=False,
                        new_columns=["chromosome", "rsid", "cm", "bp"],
                        dtypes={
                            "chromosome": pl.Int8,
                            "rsid": pl.Utf8,
                            "cm": pl.Float64,
                            "bp": pl.Int64         
                        }
                    )
        
        # Call parent DataFrame constructor
        super().__init__(data)

        self.map_len = self._calculate_genome_length()

    def _calculate_genome_length(self) -> float:
        """Calculate total genome length from the map data."""
        if len(self) == 0:
            return 0.0
        
        # Get the range for each chromosome using Polars syntax
        chrom_lengths = (
            self.group_by('chromosome')
            .agg([
                pl.col('cm').min().alias('min_cm'),
                pl.col('cm').max().alias('max_cm')
            ])
            .with_columns(
                (pl.col('max_cm') - pl.col('min_cm')).alias('length')
            )
            .select('length')
            .sum()
        )
        
        return chrom_lengths.item() 
    
    def get_col(self, col: str) -> np.ndarray:
        return self.select(col).to_numpy()
    
    @property
    def _constructor(self):
        """Ensures that operations return PlinkMap objects instead of DataFrames."""
        return PlinkMap
    
    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata during operations."""
        return self
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, genome_length: float = None):
        """Create PlinkMap from existing DataFrame."""
        instance = cls.__new__(cls)
        super(PlinkMap, instance).__init__(df)
        
        # Set custom attributes
        if genome_length is not None:
            instance.map_len = genome_length
        else:
            instance.map_len = instance._calculate_genome_length()
        
        return instance

class GeneticMap:

    def __init__(self, map_df: PlinkMap, genome_len: float):

        self.map_df = map_df

        self.genome_len = genome_len

    @classmethod
    def add_plink(cls, map_file: Union[str, Path]):

        map_df = PlinkMap(map_file)

        # Get genome length
        genome_len = map_df.map_len

        return cls(map_df, genome_len)
    
    @classmethod
    def add_plink_list(cls, map_file_list: List[Union[str, Path]]):

        map_df_list = []
        genome_len = 0
        for map_file in map_file_list:
            map_df = PlinkMap(map_file)
            genome_len += map_df.map_len
            map_df_list.append(map_df)

        return cls(pl.concat(map_df_list), genome_len)

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

            chrom_map = self.map_df.filter(pl.col("chromosome") == chrom)
        
            cm_val = chrom_map.get_column("cm").to_numpy()
            bp_val = chrom_map.get_column("bp").to_numpy()

            interp_arr[chrom_idx] = np.interp(arr[chrom_idx], bp_val, cm_val)

        return interp_arr

    def get_genome_length(self):
        return self.genome_len
