
import argparse
import sys
from pathlib import Path
from .config import PonderosaConfig

def create_parser():
    """Create argparse parser matching your config structure"""
    parser = argparse.ArgumentParser(description="PONDEROSA: Relationship inference from IBD segments")
    
    # Config file option
    parser.add_argument('--config', type=Path, help='YAML configuration file')
    
    # Files group (maps to FilesConfig)
    files_group = parser.add_argument_group('file arguments')
    files_group.add_argument('--ibd', type=Path, help='IBD segments file')
    files_group.add_argument('--fam', type=Path, help='PLINK FAM file')
    files_group.add_argument('--ibd-caller', choices=['phasedibd', 'hapibd', 'ibdseq'], 
                            default='phasedibd', help='IBD calling software', dest='ibd_caller',)
    files_group.add_argument('--ages', type=Path, help='Ages file')
    files_group.add_argument('--map', type=Path, help='Genetic map file', dest="mapf")
    files_group.add_argument('--populations', type=Path, help='Populations file')
    files_group.add_argument('--training', type=Path, help='Training models directory')
    
    # Algorithm group (maps to AlgorithmConfig)
    algo_group = parser.add_argument_group('algorithm arguments')
    algo_group.add_argument('--min-segment-length', type=float, default=3.0, 
                           help='Minimum segment length in cM')
    algo_group.add_argument('--min-total-ibd', type=float, default=50.0,
                           help='Minimum total IBD in cM')
    algo_group.add_argument('--max-gap', type=float, default=1.0,
                           help='Maximum gap for stitching segments')
    algo_group.add_argument('--population', default='pop1', help='Population identifier')
    algo_group.add_argument('--genome-length', type=float, default=3545.0,
                           help='Genome length in cM')
    
    # Output group (maps to OutputConfig)  
    output_group = parser.add_argument_group('output arguments')
    output_group.add_argument('--output', default='ponderosa_results', help='Output prefix')
    output_group.add_argument('--min-probability', type=float, default=0.5,
                             help='Minimum probability for reporting')
    output_group.add_argument('--create-plots', action='store_true', help='Generate plots')
    output_group.add_argument('--verbose', '-v', action='count', default=0,
                             help='Increase verbosity')
    output_group.add_argument("--write_training", action="store_true", help="Writes out a pickle file of the trained classifiers.")
    output_group.add_argument('--debug', action='store_true', help='Show full tracebacks')
    
    return parser

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Convert argparse Namespace to dict
        args_dict = vars(args)
        
        # Create config using your existing method
        config = PonderosaConfig.from_cli_and_yaml(args_dict)
        config.validate()
        
        # Run analysis
        from .core import run_ponderosa
        results = run_ponderosa(config)
                
    except Exception as e:
        if args.debug:
            import traceback
            print(f"Error: {e}")
            print("\nDebug traceback:")
            traceback.print_exc()
        else:
            print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()