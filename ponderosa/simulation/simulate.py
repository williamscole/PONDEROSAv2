"""
Script for managing/organizing the simulation
"""

from pathlib import Path
import gzip
from ..config import PonderosaConfig
from .config import SimulationConfig
from .setup import simulation_workspace
from .founders import pedsim_dryrun, calculate_relatedness, create_founders_file
from .pedsim import PedSim


def simulate(config: SimulationConfig) -> PonderosaConfig:
    """
    Main simulation workflow.
    
    Returns a PonderosaConfig for running PONDEROSA analysis.
    """
    
    with simulation_workspace(config) as temp_dir:
        
        vcf_ext = ".vcf.gz" if str(config.pedsim.vcf_file).endswith('.gz') else ".vcf"

        pedsim = PedSim(
            vcf_file=str(temp_dir / f"input{vcf_ext}"),  # Use dynamic extension
            def_file=str(temp_dir / "pedigree.def"),
            intf_file=str(temp_dir / "interference.txt"),
            map_file=str(temp_dir / "input.map"),
            output=str(temp_dir / "simulation"),
            executable_path=str(config.pedsim.pedsim_executable),
        )
        
        # 2. Dry run to get family structures
        dry_run_families = pedsim_dryrun(pedsim)
        
        # 3. Calculate relatedness from KING file
        relatedness_dict = calculate_relatedness(
            ibd_file=[str(config.king_file)],
            ibd_caller="king",  # or from config
            genetic_map_file_list=[]  # Add map files if needed
        )
        
        # 4. Get VCF samples
        vcf_samples = _get_vcf_samples(vcf_file=Path(pedsim.get_input("vcf")))
        
        # 5. Create founders mapping
        founders_df = create_founders_file(
            vcf_samples=vcf_samples,
            dry_run_families=dry_run_families,
            relatedness_dict=relatedness_dict,
            r=config.training.max_kinship,
            n=config.training.n_pairs_per_relationship
        )
        
        # 6. Write founders file
        founders_file = temp_dir / "founders.txt"
        founders_df.to_csv(founders_file, sep="\t", index=False, header=False)
        
        # 7. Update PedSim with founders and execute
        pedsim.update_flag("--set_founders", str(founders_file))
        pedsim.execute()
        
        # 8. Run IBD caller on simulated output
        # TODO: Implement IBD calling
        
        # 9. Create PonderosaConfig from simulation results
        ponderosa_config = _create_ponderosa_config(temp_dir, config)
        
        return ponderosa_config


def _get_vcf_samples(vcf_file: Path) -> list[str]:
    """Extract sample IDs from VCF header."""
    import gzip
    
    # Determine how to open the file
    if str(vcf_file).endswith('.gz'):
        open_func = lambda f: gzip.open(f, 'rt', encoding='utf-8')
    else:
        open_func = lambda f: open(f, 'r', encoding='utf-8')
    
    with open_func(vcf_file) as f:
        for line in f:
            if line.startswith('#CHROM'):
                # Header line found - extract sample IDs (columns 9 onward)
                columns = line.strip().split('\t')
                return columns[9:]
    
    # If we get here, no header line was found
    raise ValueError(f"No #CHROM header line found in VCF file: {vcf_file}")


def _create_ponderosa_config(temp_dir: Path, sim_config: SimulationConfig) -> PonderosaConfig:
    """Create PonderosaConfig from simulation output."""
    # TODO: Implement conversion to PonderosaConfig
    pass