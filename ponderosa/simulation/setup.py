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
    
    Creates temporary directory in the same location as output_path.
    This ensures temp files are on the same filesystem as final outputs,
    which is important for HPC environments.
    
    Args:
        config: SimulationConfig with cleanup_temp flag
            - cleanup_temp=True: Delete temp directory after completion
            - cleanup_temp=False: Preserve temp directory for inspection
    """
    
    # Determine where to create temp directory (same parent as output_path)
    output_path = Path(config.output_path)
    output_dir = output_path.parent if output_path.parent != Path('.') else Path.cwd()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config.cleanup_temp:
        # Use TemporaryDirectory for auto-cleanup
        temp_dir_obj = tempfile.TemporaryDirectory(
            prefix="ponderosa_sim_",
            dir=output_dir  # Create in same directory as output
        )
        temp_path = Path(temp_dir_obj.name)
    else:
        # Use mkdtemp for persistent directory
        temp_path = Path(tempfile.mkdtemp(
            prefix="ponderosa_sim_",
            dir=output_dir
        ))
        temp_dir_obj = None
    
    try:
        # Setup
        _symlink_files(temp_path, config)
        _validate_simulation_files(temp_path, config)
        
        print(f"Temporary workspace: {temp_path}")
        print(f"Final outputs will be saved to: {output_path}")
        
        yield temp_path
        
    finally:
        if config.cleanup_temp and temp_dir_obj:
            print(f"Cleaning up temporary directory: {temp_path}")
            temp_dir_obj.cleanup()
        elif not config.cleanup_temp:
            print(f"Temporary directory preserved at: {temp_path}")
            print(f"Remember to manually delete when done!")
            

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
    
    # Check for VCF (either .vcf or .vcf.gz)
    vcf_found = False
    if (temp_dir / "input.vcf").exists():
        vcf_found = True
    elif (temp_dir / "input.vcf.gz").exists():
        vcf_found = True
    
    if not vcf_found:
        raise FileNotFoundError(
            f"Required VCF file not found: expected input.vcf or input.vcf.gz in {temp_dir}"
        )
    
    # Check other required files
    required = ["pedigree.def"]
    
    for filename in required:
        file_path = temp_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required simulation file not found: {filename} "
                f"(expected at {file_path})"
            )