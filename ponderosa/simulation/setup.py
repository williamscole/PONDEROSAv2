"""
Script for setting up the simulation.

Should do the following:

- Create a temporary directory to write to
- symlink necessary files to the directory such that the file names are simple
    - input.vcf
- Ensure that simmap, inft arguments are given and files exist
- Ensure that def file exists
"""

"""
Script for setting up the simulation.
"""

import tempfile
from pathlib import Path
from contextlib import contextmanager
import shutil
from .config import SimulationConfig

@contextmanager
def simulation_workspace(config: SimulationConfig):
    """
    Context manager for simulation workspace with automatic cleanup.
    
    Usage:
        with simulation_workspace(config) as temp_dir:
            # Do simulation work
            pedsim.execute()
    """
    
    if config.temp_dir:
        temp_path = Path(config.temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        temp_dir_obj = None
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="ponderosa_sim_")
        temp_path = Path(temp_dir_obj.name)
    
    try:
        # Setup
        _symlink_files(temp_path, config)
        _validate_simulation_files(temp_path, config)
        
        yield temp_path
        
    finally:
        # Cleanup
        if config.cleanup_temp:
            if temp_dir_obj:
                temp_dir_obj.cleanup()
            elif config.temp_dir:
                shutil.rmtree(temp_path, ignore_errors=True)


def _symlink_files(temp_dir: Path, config: SimulationConfig):
    """Create symlinks to input files with simple names."""
    
    # Determine VCF extension based on source file
    vcf_ext = ".vcf.gz" if str(config.pedsim.vcf_file).endswith('.gz') else ".vcf"
    
    symlinks = {
        f"input{vcf_ext}": config.pedsim.vcf_file,
        "input.map": config.pedsim.simmap_file,
        "interference.txt": config.pedsim.interference_file,
        "pedigree.def": config.pedsim.def_file,
    }
    
    for simple_name, source_path in symlinks.items():
        if source_path and source_path.exists():
            target = temp_dir / simple_name
            target.symlink_to(source_path.resolve())


def _validate_simulation_files(temp_dir: Path, config: SimulationConfig):
    """Ensure that required simulation files exist."""
    
    required = ["input.vcf", "input.map", "pedigree.def"]
    
    for filename in required:
        file_path = temp_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required simulation file not found: {filename} "
                f"(expected at {file_path})"
            )