"""
Script for setting up the simulation workspace.

This module provides:
- Temporary workspace creation with configurable cleanup
- File symlinking for simplified paths
- Validation of required simulation files
"""

import tempfile
from pathlib import Path
from contextlib import contextmanager
import shutil
import logging
from .config import SimulationConfig

logger = logging.getLogger(__name__)


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
    
    Yields:
        Path: Path to the temporary workspace directory
        
    Example:
        >>> config = SimulationConfig.from_yaml("sim_config.yaml")
        >>> with simulation_workspace(config) as temp_dir:
        ...     # Do simulation work in temp_dir
        ...     pedsim.execute()
        >>> # Temp dir automatically cleaned up if cleanup_temp=True
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
        logger.info(f"Creating simulation workspace at: {temp_path}")
        _symlink_files(temp_path, config)
        _validate_simulation_files(temp_path, config)
        
        logger.info(f"Temporary workspace: {temp_path}")
        logger.info(f"Final outputs will be saved to: {output_path}")
        
        yield temp_path
        
    finally:
        if config.cleanup_temp and temp_dir_obj:
            logger.info(f"Cleaning up temporary directory: {temp_path}")
            temp_dir_obj.cleanup()
        elif not config.cleanup_temp:
            logger.info(f"Temporary directory preserved at: {temp_path}")
            logger.info(f"Remember to manually delete when done!")


def _symlink_files(temp_dir: Path, config: SimulationConfig):
    """
    Create symlinks to input files with simple, standardized names.
    
    Args:
        temp_dir: Temporary workspace directory
        config: SimulationConfig containing file paths
        
    Creates symlinks:
        - input.vcf or input.vcf.gz -> VCF file
        - input.map -> Genetic map file
        - interference.tsv -> Crossover interference file
        - pedigree.def -> Pedigree definition file
    
    Raises:
        FileNotFoundError: If required files are missing
    """
    
    # Determine VCF extension based on source file
    vcf_ext = ".vcf.gz" if str(config.pedsim.vcf_file).endswith('.gz') else ".vcf"
    
    # Define symlinks with their importance level
    symlinks = {
        f"input{vcf_ext}": {
            "source": config.pedsim.vcf_file,
            "required": True,
            "description": "VCF file with founder genotypes"
        },
        "input.map": {
            "source": config.pedsim.simmap_file,
            "required": True,
            "description": "genetic map file"
        },
        "interference.tsv": {
            "source": config.pedsim.interference_file,
            "required": True,
            "description": "crossover interference parameters"
        },
        "pedigree.def": {
            "source": config.pedsim.def_file,
            "required": True,
            "description": "pedigree definition file"
        },
    }
    
    created_count = 0
    missing_required = []
    
    for simple_name, info in symlinks.items():
        source_path = info["source"]
        is_required = info["required"]
        description = info["description"]
        
        if source_path is None:
            if is_required:
                missing_required.append(
                    f"  • {simple_name}: No source path specified ({description})"
                )
                logger.error(f"Missing required file: {simple_name} - {description}")
            else:
                logger.debug(f"Skipping optional {simple_name}: source path is None")
            continue
        
        if not source_path.exists():
            if is_required:
                missing_required.append(
                    f"  • {simple_name}: Source file not found at {source_path}\n"
                    f"    ({description})"
                )
                logger.error(f"Required file not found: {source_path}")
            else:
                logger.warning(f"Optional file not found: {source_path} ({description})")
            continue
        
        # Create the symlink
        target = temp_dir / simple_name
        try:
            target.symlink_to(source_path.resolve())
            logger.info(f"✓ Created symlink: {simple_name} -> {source_path}")
            created_count += 1
        except OSError as e:
            error_msg = f"Failed to create symlink {simple_name} -> {source_path}: {e}"
            if is_required:
                missing_required.append(f"  • {simple_name}: {error_msg}")
            logger.error(error_msg)
    
    # Report results
    if missing_required:
        error_message = (
            f"Failed to create workspace: {len(missing_required)} required file(s) missing:\n" +
            "\n".join(missing_required) +
            "\n\nPlease ensure all required files exist before running simulation.\n"
            "Check your configuration and run config.validate() to diagnose issues."
        )
        raise FileNotFoundError(error_message)
    
    logger.info(f"Successfully created {created_count} symlinks in workspace")


def _validate_simulation_files(temp_dir: Path, config: SimulationConfig):
    """
    Ensure that required simulation files exist in the workspace.
    
    Args:
        temp_dir: Temporary workspace directory
        config: SimulationConfig for context
        
    Raises:
        FileNotFoundError: If required files are missing
    """
    
    logger.info("Validating simulation workspace...")
    
    missing_files = []
    
    # Check for VCF (either .vcf or .vcf.gz)
    vcf_found = False
    vcf_path = None
    for vcf_name in ["input.vcf.gz", "input.vcf"]:
        vcf_path = temp_dir / vcf_name
        if vcf_path.exists():
            vcf_found = True
            logger.info(f"✓ Found VCF file: {vcf_name}")
            break
    
    if not vcf_found:
        missing_files.append(
            "  • VCF file (input.vcf or input.vcf.gz): Required for founder genotypes"
        )
    
    # Check required files
    required_files = {
        "pedigree.def": "pedigree definitions for simulation",
        "input.map": "genetic map for recombination",
        "interference.tsv": "crossover interference parameters",
    }
    
    for filename, description in required_files.items():
        file_path = temp_dir / filename
        if file_path.exists():
            logger.info(f"✓ Found {filename}")
        else:
            missing_files.append(
                f"  • {filename}: {description}"
            )
    
    # If any required files are missing, raise error
    if missing_files:
        error_message = (
            f"Workspace validation failed: {len(missing_files)} required file(s) missing:\n" +
            "\n".join(missing_files) +
            f"\n\nWorkspace location: {temp_dir}\n"
            "\nThis usually means:\n"
            "  1. Files were not found during auto-detection\n"
            "  2. Files do not exist at specified paths\n"
            "  3. Config validation was not run (try config.validate())\n"
            "\nCheck your configuration and ensure all files exist."
        )
        raise FileNotFoundError(error_message)
    
    logger.info("Workspace validation complete - all required files present")