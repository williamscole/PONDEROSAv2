"""
Configuration management for PONDEROSA genetic relationship simulation.

This module provides configuration for simulating relative pairs using ped-sim
to generate training data for relationship classifiers.

Classes:
    PedSimConfig: Configuration for ped-sim execution
    TrainingConfig: Configuration for training data requirements
    SimulationConfig: Main simulation configuration container
    
Example:
    >>> config = SimulationConfig.from_yaml("simulation.yaml")
    >>> config.validate()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union, Any, List
import yaml
import logging
import sys

__all__ = ['PedSimConfig', 'TrainingConfig', 'SimulationConfig']

logger = logging.getLogger(__name__)


@dataclass
class PedSimConfig:
    """Configuration for ped-sim execution."""
    
    # ped-sim paths and files
    pedsim_path: Path  # Directory containing ped-sim executable
    vcf_file: Path  # Input VCF for founder genotypes
    simmap_file: Optional[Path] = None  # Sex-specific recombination map
    interference_file: Optional[Path] = None  # Interference file
    def_file: Optional[Path] = None  # Pedigree definition file
    
    # ped-sim parameters
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Set default paths for simmap, interference, and def files if not provided."""
        
        # Default simmap file location
        if self.simmap_file is None:
            default_simmap = self.pedsim_path / "refined_mf.simmap"
            if default_simmap.exists():
                self.simmap_file = default_simmap
                logger.info(f"Using default simmap file: {default_simmap}")
        
        # Default interference file location  
        if self.interference_file is None:
            default_intf = self.pedsim_path / "interfere" / "nu_p_campbell.tsv"
            if default_intf.exists():
                self.interference_file = default_intf
                logger.info(f"Using default interference file: {default_intf}")
        
        # Default def file location
        if self.def_file is None:
            # This would be set relative to the ponderosa package
            default_def = Path(__file__).parent.parent / "config" / "default.def"
            if default_def.exists():
                self.def_file = default_def
                logger.info(f"Using default def file: {default_def}")
    
    @property
    def pedsim_executable(self) -> Path:
        """Path to ped-sim executable."""
        return self.pedsim_path / "ped-sim"
    
    def validate(self) -> None:
        """Validate ped-sim configuration."""
        
        # Check ped-sim executable exists
        if not self.pedsim_executable.exists():
            raise FileNotFoundError(f"ped-sim executable not found: {self.pedsim_executable}")
        
        # Check required files exist
        required_files = [self.vcf_file]
        for file_path in required_files:
            if file_path and not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Check optional files if provided
        optional_files = [self.simmap_file, self.interference_file, self.def_file]
        for file_path in optional_files:
            if file_path and not file_path.exists():
                logger.warning(f"Optional file not found: {file_path}")


@dataclass
class TrainingConfig:
    """Configuration for training data generation."""
    
    # Simulation parameters
    n_pairs_per_relationship: int = 100  # Number of pairs to simulate for each relationship type
    
    # Founder selection parameters
    max_kinship: float = 0.05  # Maximum kinship between founders in a family
    
    def validate(self) -> None:
        """Validate training configuration."""
        
        # Check positive pair count
        if self.n_pairs_per_relationship <= 0:
            raise ValueError(f"n_pairs_per_relationship must be positive, got {self.n_pairs_per_relationship}")
        
        # Check kinship threshold
        if not 0 <= self.max_kinship <= 0.5:
            raise ValueError("max_kinship must be between 0 and 0.5")


@dataclass  
class SimulationConfig:
    """Main simulation configuration container."""
    
    # Required components
    pedsim: PedSimConfig
    king_file: Path  # KING output file for founder selection
    
    # Optional components
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Processing settings
    ibd_caller: str = "hap-ibd.sh"  # Bash script for IBD calling
    output_path: str = "ponderosa_simulation"  # Output directory/prefix
    cleanup_temp: bool = True  # Whether to clean up temporary files
    
    @classmethod
    def print_help_and_exit(cls):
        """Print simulation configuration help and exit."""
        help_text = """
PONDEROSA Simulation Configuration Help
=====================================

To run simulations, create a YAML file with the following structure:

# Required parameters
pedsim:
  pedsim_path: "/path/to/ped-sim"           # Directory containing ped-sim executable
  vcf_file: "/path/to/input.vcf"            # Input VCF with founder genotypes
  
king_file: "/path/to/king.seg"              # KING output for founder selection

# Optional parameters (with defaults shown)
pedsim:
  simmap_file: null                         # Auto-found in ped-sim directory if null
  interference_file: null                   # Auto-found in ped-sim directory if null  
  def_file: null                            # Uses ponderosa/config/default.def if null
  random_seed: null                         # Random seed if not specified

training:
  n_pairs_per_relationship: 100             # Number of pairs per relationship type
  max_kinship: 0.05                         # Max kinship between founders in a family

# Processing settings (optional)
ibd_caller: "hap-ibd.sh"                     # Bash script name for IBD calling
output_path: "ponderosa_simulation"         # Output directory/prefix
temp_dir: null                              # Uses system temp if null
cleanup_temp: true                          # Clean temporary files when done

Example YAML file:
---
pedsim:
  pedsim_path: "/home/user/ped-sim"
  vcf_file: "/data/founders.vcf"
  random_seed: 12345

king_file: "/data/kinship.seg"

training:
  n_pairs_per_relationship: 150
  max_kinship: 0.03

output_path: "my_simulation"
cleanup_temp: false

Usage:
  python ponderosa.py simulate --config simulation.yaml
        """
        print(help_text)
        sys.exit(0)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "SimulationConfig":
        """Load simulation configuration from YAML file."""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Simulation configuration file not found: {yaml_path}")
        
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimulationConfig":
        """Create simulation configuration from dictionary."""
        
        # Handle pedsim config
        pedsim_dict = config_dict.get("pedsim", {})
        
        # Convert string paths to Path objects
        path_fields = {'pedsim_path', 'simmap_file', 'interference_file', 'vcf_file', 'def_file'}
        for key, value in pedsim_dict.items():
            if key in path_fields and value is not None:
                pedsim_dict[key] = Path(value)
        
        pedsim_config = PedSimConfig(**pedsim_dict)
        
        # Handle training config
        training_dict = config_dict.get("training", {})
        training_config = TrainingConfig(**training_dict)
        
        # Handle main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in {"pedsim", "training"}}
        
        # Convert path fields
        if "king_file" in main_config:
            main_config["king_file"] = Path(main_config["king_file"])
        if "temp_dir" in main_config and main_config["temp_dir"]:
            main_config["temp_dir"] = Path(main_config["temp_dir"])
        
        return cls(
            pedsim=pedsim_config,
            training=training_config,
            **main_config
        )
    
    def validate(self) -> None:
        """Validate the entire simulation configuration."""
        
        # Validate nested configs
        self.pedsim.validate()
        self.training.validate()
        
        # Check king file exists
        if not self.king_file.exists():
            raise FileNotFoundError(f"KING file not found: {self.king_file}")
        
        # Check if we have reasonable number of relationships for training
        total_pairs = len(self.training.relationship_types) * self.training.n_pairs_per_relationship
        if total_pairs < 50:
            logger.warning(f"Only {total_pairs} total pairs specified - may not be sufficient for training")
        
        logger.info("Simulation configuration validation passed")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for YAML export."""
        return {
            "pedsim": {
                "pedsim_path": str(self.pedsim.pedsim_path),
                "vcf_file": str(self.pedsim.vcf_file),
                "simmap_file": str(self.pedsim.simmap_file) if self.pedsim.simmap_file else None,
                "interference_file": str(self.pedsim.interference_file) if self.pedsim.interference_file else None,
                "def_file": str(self.pedsim.def_file) if self.pedsim.def_file else None,
                "random_seed": self.pedsim.random_seed,
            },
            "king_file": str(self.king_file),
            "training": {
                "n_pairs_per_relationship": self.training.n_pairs_per_relationship,
                "relationship_types": self.training.relationship_types,
                "max_kinship": self.training.max_kinship,
            },
            "ibd_caller": self.ibd_caller,
            "output_path": self.output_path,
            "temp_dir": str(self.temp_dir) if self.temp_dir else None,
            "cleanup_temp": self.cleanup_temp,
        }
    
    def save_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        
        logger.info(f"Simulation configuration saved to {output_path}")