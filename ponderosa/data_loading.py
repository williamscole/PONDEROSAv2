from abc import ABC, abstractmethod
import polars as pl
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Any, Optional, Union, Dict
import numpy as np

from .config import FilesConfig, AlgorithmConfig
from .ibd_tools import IBD, ProcessSegments, Features

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


##### IBD loading methods #####
# TODO: implement interpolation of genetic map (it can currently be added to the class when init, but need to update scan_file or add another function to do the interp)
class IBDLoader(ABC):
    """Abstract contract for all IBD loading classes."""
    def __init__(self, min_segment_length: float = 5.0, min_total_ibd: float = 100.0, genetic_map: Optional['GeneticMap'] = None):

        self.min_segment_length = min_segment_length
        self.min_total_ibd = min_total_ibd
        self.genetic_map = genetic_map

    @abstractmethod
    def scan_file(self, file_path: str) -> pl.LazyFrame:
        """
        Return a Polars LazyFrame with standardized columns.
        """
        pass

    @abstractmethod
    def custom_process(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        pass

    def filter_pairs(self, df: Union[pl.DataFrame, pl.LazyFrame], min_total_ibd):

        if isinstance(df, pl.DataFrame):
            lazy_df = df.lazy()
        else:
            lazy_df = df

        pairs = (
            lazy_df
            .group_by(['id1', 'id2'])
            .agg([
                pl.col('length_cm').sum().alias('total_ibd'),
                pl.count().alias('n_segments')
            ])
            .filter(pl.col('total_ibd') >= min_total_ibd)
            .collect()  # Only collect the small pairs summary
        )

        good_pairs = set(zip(pairs['id1'], pairs['id2']))

        filtered_segments = (
            lazy_df
            .filter(
                pl.struct(['id1', 'id2'])
                .map_elements(lambda x: (x['id1'], x['id2']) in good_pairs, return_dtype=pl.Boolean)
            )
            .collect()  # Collect only the filtered segments
        )

        return filtered_segments
 
    def load_filtered_segments(self, file_path: str, **process_kwargs) -> pl.DataFrame:
        """Loads IBD data from a source and returns a polars DataFrame."""

        # Keep lazy as long as possible
        segments_lazy = (
            self.scan_file(file_path)
            .filter(pl.col('length_cm') >= self.min_segment_length)
        )

        filtered_segments = self.filter_pairs(segments_lazy, self.min_segment_length)

        return self.custom_process(filtered_segments, **process_kwargs)
        

class PhasedIBDLoader(IBDLoader):
    """Loader for phasedibd output format."""
    
    def scan_file(self, file_path: str) -> pl.LazyFrame:
        """
        Scan phasedibd output file.
        
        phasedibd format has columns:
        id1, id2, chromosome, start_cm, end_cm, id1_haplotype, id2_haplotype
        """
        df = pl.scan_csv(
            file_path,
            separator='\t',
            dtypes={
                'id1': pl.Utf8,
                'id2': pl.Utf8,
                'chromosome': pl.Int8,
                'start_cm': pl.Float64,
                'end_cm': pl.Float64,
                'id1_haplotype': pl.Int8,
                'id2_haplotype': pl.Int8
            }
        )

        df = df.with_columns(
        (pl.col('end_cm') - pl.col('start_cm')).alias('length_cm')
        )
        
        return df.select([
                'id1', 'id2', 'chromosome', 'start_cm', 'end_cm', 
                'id1_haplotype', 'id2_haplotype', 'length_cm'
            ])
    
    def custom_process(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return df

class HapIBDLoader(IBDLoader):

    def scan_file(self, file_path: str) -> pl.LazyFrame:
        
        # Read the 8-column HapIBD file (no header)
        df = pl.scan_csv(
            file_path,
            separator='\t',
            has_header=False,
            new_columns=[
                'id1', 'id1_haplotype', 'id2', 'id2_haplotype', 
                'chromosome', 'start_bp', 'end_bp', 'length_cm'
            ],
            dtypes={
                'id1': pl.Utf8,
                'id1_haplotype': pl.Int8,
                'id2': pl.Utf8, 
                'id2_haplotype': pl.Int8,
                'chromosome': pl.Int8,
                'start_bp': pl.Int64,
                'end_bp': pl.Int64,
                'length_cm': pl.Float64
            }
        )

        return df.with_columns([
                (pl.col('id1_haplotype') - 1).alias('id1_haplotype'),
                (pl.col('id2_haplotype') - 1).alias('id2_haplotype')
            ]).select([
                'id1', 'id2', 'chromosome', 'start_bp', 'end_bp',
                'id1_haplotype', 'id2_haplotype', 'length_cm'
            ])
    
    def custom_process(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """Convert bp coordinates to cm using genetic map interpolation."""
        genetic_map = kwargs["genetic_map"]
        
        # Convert start_bp to start_cm
        start_bp_array = df.get_column("start_bp").to_numpy()
        chrom_array = df.get_column("chromosome").to_numpy()
        start_cm_array = genetic_map.interp(start_bp_array, chrom_array)
        
        # Convert end_bp to end_cm  
        end_bp_array = df.get_column("end_bp").to_numpy()
        end_cm_array = genetic_map.interp(end_bp_array, chrom_array)
        
        # Add the new cm columns and drop bp columns
        result = df.with_columns([
            pl.Series("start_cm", start_cm_array),
            pl.Series("end_cm", end_cm_array)
        ]).drop(["start_bp", "end_bp"])
        
        return result
            

        
def load_ibd_from_file(file_paths: List[Path], ibd_caller: str, min_segment_length: float, min_total_ibd: float, to_pandas: bool = False) -> IBD:

    ibd_caller_dict = {
        "phasedibd": PhasedIBDLoader,
        "hap-ibd": HapIBDLoader
    }

    loader_class = ibd_caller_dict[ibd_caller]

    single_file = len(file_paths) == 1

    # If multiple files, cannot filter based on min_total_ibd (since loading only for each chromosome)
    loader = loader_class(min_segment_length, min_total_ibd if single_file else min_segment_length)

    ibd_segments_list = []

    for file_path in file_paths:
        ibd_segments_list.append(loader.load_filtered_segments(file_path))

    if single_file:
        ibd_segments = ibd_segments_list[0]
    # Multiple files added; must concagt and then filter on total ibd
    else:
        ibd_segments = pl.concat(ibd_segments_list)

        ibd_segments = loader.filter_pairs(ibd_segments, min_total_ibd)
    
    return ibd_segments.to_pandas() if to_pandas else ibd_segments


####### Individual level methods #######
class Individuals:
    """
    Holds individual-level data, including:
    father: str
    mother: str
    child: list[str]
    sex: 0=none, 1=male, 2=female
    age: float
    """

    FATHER = 0
    MOTHER = 1
    CHILDREN = 2
    SEX = 3
    AGE = 4

    def __init__(self):

        self.default_values = {self.FATHER: "0",
                               self.MOTHER: "0",
                               self.CHILDREN: [],
                               self.SEX: 0,
                               self.AGE: np.nan}

        self.n_values = len(self.default_values)

        self.default_missing_parent = ["0"]

        self.individuals = dict()

    def _get_default(self):
        default_vals = []
        for i in range(self.n_values):
            val = self.default_values[i]
            if val == []:
                default_vals.append([])
            else:
                default_vals.append(val)
        return default_vals

    def _add_iid(self, iid: str, val: Any, index: int):
        
        if iid not in self.individuals:
            self.individuals[iid] = self._get_default()

        self.individuals[iid][index] = val

    def _add_child(self, parent_iid: str, child_iid: str, father: bool):

        if parent_iid not in self.default_missing_parent:

            self._add_iid(parent_iid, 1 if father else 2, self.SEX)

            self.individuals[parent_iid][self.CHILDREN].append(child_iid)

    def add_fam_file(self, fam_file: str):

        fam_df = pd.read_csv(fam_file, sep="\\s+",
                             names=["fid", "iid", "father", "mother", "sex", "pheno"],
                             dtype={"iid": str, "father": str, "mother": str, "sex": int})

        for row in fam_df.itertuples():
            self._add_iid(row.iid, row.father, self.FATHER)
            self._add_iid(row.iid, row.mother, self.MOTHER)
            self._add_iid(row.iid, row.sex, self.SEX)
            self._add_child(row.father, row.iid, father=True)
            self._add_child(row.mother, row.iid, father=False)

    def add_age_file(self, age_file: str):

        age_df = pd.read_csv(age_file, sep="\\s+",
                             names=["iid", "age"],
                             dtype={"iid": str, "age": float})

        assert (age_df["age"] > 1000).all() or (age_df["age"] <= 1000).all()

        # Years of birth are given
        if (age_df["age"] > 1000).all():
            cur_year = datetime.now().year
            age_df["age"] = cur_year - age_df["age"]

        for row in age_df.itertuples():
            self._add_iid(row.iid, row.age, self.AGE)

    def get_ids(self) -> list:
        return list(self.individuals.keys())

    def _get_value(self, index: int, iid: str) -> Any:
        return self.individuals[iid][index]

    def _get_values(self, index_list: List[int], iid: str) -> List[Any]:
        return self.individuals[iid][index_list]

    def _get_value_from_list(self, index: int, iid_list: List[str] = None) -> List[Any]:
        return [self.individuals[iid][index] for iid in iid_list]        

    def _get_values_from_list(self, index_list: List[int], iid_list: List[str] = None) -> List[List[Any]]:
        if iid_list:
            return [self.individuals[iid][index_list] for iid in iid_list]

    def get_age(self, iid: str) -> float:
        return self._get_value(self.AGE, iid)

    def get_sex(self, iid: str) -> int:
        return self._get_value(self.SEX, iid)

    def get_father(self, iid: str) -> str:
        return self._get_value(self.FATHER, iid)

    def get_mother(self, iid: str) -> str:
        return self._get_value(self.MOTHER, iid)

    def get_parents(self, iid: str) -> List[str]:
        return self._get_values([self.FATHER, self.MOTHER], iid)

    def get_child(self, iid: str) -> List[str]:
        return self._get_value(self.CHILDREN, iid)

    def _format_return(self, iid_list: np.array, *args, output_style: str = "zip"):

        if output_style == "zip":
            return tuple(list([iid_list]) + list(args))

        output_data = []
        for index in range(iid_list.shape[0]):

            iid = iid_list[index]

            if output_style == "flatten":
                entry = [iid]

                for arr in args:
                    entry.append(arr[index])

                output_data.append(entry)

            if output_style == "expand":

                for arr in args:
                    output_data.append([iid, arr[index]])

        return np.array(output_data, dtype=object)


    def retrieve_data(self, *args, iid_list: np.array = None, output_style: str = "zip"):
        if iid_list:
            iid_list = np.array(iid_list)
        else:
            iid_list = np.array(list(self.individuals.keys()))

        arrs = []

        for arg in args:

            assert hasattr(self, arg)

            index = getattr(self, arg)

            arrs.append(np.array(self._get_value_from_list(index, iid_list)))

        return self._format_return(iid_list, *arrs, output_style=output_style)

    def get_default_missing_parent(self) -> list:
        return self.default_missing_parent


####### Pair-level methods #######
class Pairs:

    def __init__(self, ibd_feature_df: pl.DataFrame):

        data = ibd_feature_df.drop(["id1", "id2"]).to_numpy().copy()

        self.pair_to_index = {}
        self.index_to_pair = {}

        for index, row in enumerate(ibd_feature_df[["id1", "id2"]].iter_rows()):

            pair = self._get_pair(*row)

            swapped = pair[0] != row[0]

            if swapped: # Swap the h1, h2 values
                data[index, [Features.H1, Features.H2]] = data[index, [Features.H2, Features.H1]]
                data[index, [Features.H1_ERR, Features.H2_ERR]] = data[index, [Features.H2_ERR, Features.H1_ERR]]


            self.pair_to_index[pair] = index
            self.index_to_pair[index] = pair
            
        nan_array = np.full(data.shape[1], np.nan)
        self.ibd_data = np.vstack([data, nan_array])

        self.max_index = index + 1

    @classmethod
    def from_segment_df(cls, ibd_segments: IBD, max_gap: float = 1.0):

        ibd_feature_df = ibd_segments.group_by("id1", "id2").map_groups(lambda group: cls._compute_pair(cls, group, max_gap))

        return cls(ibd_feature_df, max_gap)

    def n_pairs(self):
        return self.max_index

    def get_pair_dict(self, index_to_pair: bool = False):
        return self.index_to_pair if index_to_pair else self.pair_to_index
        
    def _pair_order(self, id1: str, id2: str) -> bool:
        return id1 < id2

    def _get_pair(self, id1: str, id2: str) -> Tuple[str, str]:
        return (id1, id2) if self._pair_order(id1, id2) else (id2, id1)

    def _get_pair_idx(self, id1: str, id2: str) -> int:
        pair = self._get_pair(id1, id2)

        return self.pair_to_index.get(pair, self.max_index)

    def _get_pair_data(self, id1: str, id2: str, val: str) -> float:
        if not self._pair_order(id1, id2): # Out of order
            val = Features.get_swappable(val)

        val_index = Features.get_index(val)
        
        return self.ibd_data[self._get_pair_idx(id1, id2), val_index]

    def _format_return(self, pair_list: np.array, *args, output_style: str = "zip"):

        if output_style == "zip":
            return tuple(args)

        output_data = []
        for index in range(pair_list.shape[0]):

            pair = pair_list[index]

            if output_style == "flatten":
                entry = []

                for arr in args:
                    entry.append(arr[index])

                output_data.append(entry)

            if output_style == "expand":
                for arr in args:
                    output_data.append([*pair, arr[index]])

        if output_style == "expand":
            return np.array(output_data, dtype=object)

        return np.array(output_data)

    def get_pair_data_from(self, pair_list: List[Tuple[str, str]], *args, output_style: str = "zip"):

        if len(pair_list) == 0:
            pair_list = np.array(sorted([pair for pair in self.pair_to_index]))

        arrs = []

        assert isinstance(pair_list, np.ndarray)

        for arg in args:

            assert hasattr(Features, arg)

            val_index = getattr(Features, arg)

            arr = np.array([self._get_pair_data(*pair, arg) for pair in pair_list])

            arrs.append(arr)

        return self._format_return(pair_list, *arrs, output_style=output_style)

    def _index(self, attr):
        return FEATURE_DICT[attr]

    def _compute_pair(self, pair_df: pl.DataFrame, max_gap: float = 1.0) -> pl.DataFrame:

        arr = ProcessSegments.ibd_sharing_stats(pair_df, max_gap)

        id1_val = pair_df["id1"][0]
        id2_val = pair_df["id2"][0]

        columns = ["IBD1", "IBD2", "H1", "H2", "HTOT", "N", "N2", "H1_ERR", "H2_ERR", "HTOT_ERR"]

        data = {col: [arr[Features.get_index(col)]] for col in Features.get_feature_names()}
        data["id1"] = [id1_val]
        data["id2"] = [id2_val]

        return pl.DataFrame(data)


def load_individuals(config: FilesConfig) -> Individuals:

    individuals = Individuals()

    # Required
    individuals.add_fam_file(config.fam)

    # Optional
    if config.ages:
        individuals.add_age_file(config.ages)

    return individuals


def load_pairs(files: FilesConfig, alg_args: AlgorithmConfig) -> Pairs:

    # Load the IBD segments
    ibd_segments = load_ibd_from_file(files.ibd, files.ibd_caller, alg_args.min_segment_length, alg_args.min_total_ibd)

    # Add the IBD segments to the pairs
    pairs = Pairs.from_segment_df(ibd_segments)
    
    return pairs







        






