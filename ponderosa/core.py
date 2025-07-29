
from .config import PonderosaConfig, OutputConfig
from .data_loading import load_individuals, load_pairs
from .pedigree import build_pedigree, PedigreeHierarchy
from .classifiers import train_load_classifiers, run_inference


def run_ponderosa(config: PonderosaConfig):
    # Load the hierarchy
    hierarchy = PedigreeHierarchy.from_yaml()

    # Loads individual-level data
    individuals = load_individuals(config.files)

    # Loads pairwise data, including IBD information
    pairs = load_pairs(config.files, config.algorithm)

    # Finds all relationships in the dataset
    registry = build_pedigree(individuals, pairs, hierarchy)

    # Train/write or load classifiers
    classifiers = train_load_classifiers(registry,
                                         pairs,
                                         config.files.training,
                                         config.output.output)

    # Use the classifiers and compute posterior probability
    matrix_hierarchy = run_inference(pairs, classifiers, hierarchy)

    # Write out results


