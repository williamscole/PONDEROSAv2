# PONDEROSA

PONDEROSA is a Python tool for inferring genetic relationships between individuals using Identity By Descent (IBD) segments. The tool uses machine learning classifiers trained on IBD sharing patterns to distinguish between different degrees of biological relationships.

## Overview

PONDEROSA analyzes IBD segments to infer relationships between pairs of individuals. It supports multiple IBD calling software outputs and provides flexible configuration through both command-line arguments and YAML configuration files.

## Installation

### Prerequisites

PONDEROSA requires Python 3.13+ and the following dependencies:
- pandas
- polars  
- numpy
- scikit-learn
- pyyaml
- networkx

### Environment Setup

The recommended installation method is using conda with the provided environment file:

```bash
conda env create -f environment.yml
conda activate ponderosa_v2
```

## Usage

### Basic Command Line Usage

```bash
python -m ponderosa.cli [options]
```

### Using Configuration Files

PONDEROSA supports YAML configuration files for complex analyses:

```bash
python -m ponderosa.cli --config config.yaml
```

## Command Line Arguments

### File Arguments

These arguments specify input and output files:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--config` | Path | No | YAML configuration file |
| `--ibd` | Path | Yes | IBD segments file |
| `--fam` | Path | Yes | PLINK FAM file with individual information |
| `--ibd-caller` | Choice | Yes | IBD calling software: `phasedibd`, `hapibd` |
| `--map` | Path | Yes* | Genetic map file for coordinate conversion |
| `--ages` | Path | No | File containing age information for individuals |
| `--priors` | Path | No | File specifying relationship priors (e.g., age-based priors) |
| `--populations` | Path | No | Population assignment file |
| `--training` | Path | No | Directory containing pre-trained models |

*Unless your IBD caller outputs the segment length in cM


### Algorithm Arguments

These control the relationship inference algorithm:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--min-segment-length` | Float | 3.0 | Minimum IBD segment length in centiMorgans (cM) |
| `--min-total-ibd` | Float | 50.0 | Minimum total IBD sharing in cM for a pair to be analyzed |
| `--max-gap` | Float | 1.0 | Maximum gap in cM for stitching adjacent segments |
| `--population` | String | "pop1" | Population identifier for analysis |
| `--genome-length` | Float | 3545.0 | Total genome length in cM |

### Output Arguments

These control output format and verbosity:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output` | String | "ponderosa_results" | Output file prefix |
| `--min-probability` | Float | 0.5 | Minimum probability threshold for reporting relationships |
| `--create-plots` | Flag | False | Generate visualization plots |
| `--verbose`, `-v` | Count | 0 | Increase verbosity (can be used multiple times: `-v`, `-vv`, `-vvv`) |
| `--write_training` | Flag | False | Write trained classifiers to pickle file |
| `--debug` | Flag | False | Show full error tracebacks |

## Configuration File Format

PONDEROSA supports YAML configuration files with the following structure:

```yaml
# File inputs
files:
  ibd: "path/to/ibd_segments.txt"
  fam: "path/to/individuals.fam"
  ibd_caller: "phasedibd"
  ages: "path/to/ages.txt"                    # Optional
  map: "path/to/genetic.map"                  # Optional
  priors: "path/to/priors.yaml"               # Optional
  populations: "path/to/populations.txt"      # Optional
  training: "path/to/trained_models/"         # Optional

# Algorithm parameters  
algorithm:
  min_segment_length: 3.0
  min_total_ibd: 50.0
  max_gap: 1.0
  population: "pop1"
  genome_length: 3545.0

# Output settings
output:
  output: "my_analysis_results"
  min_probability: 0.5
  write_readable: true
  verbose: true
  write_training: false
```

## Input File Formats

### IBD Segments File

The format depends on the IBD caller specified:

#### PhaseIBD Format
```
id1    id2    chromosome    start_cm    end_cm    id1_haplotype    id2_haplotype
IND1   IND2   1             10.5        25.3      1                1
IND1   IND3   1             30.1        45.7      2                1
```

#### HapIBD Format  
```
id1    id2    chromosome    start_bp    end_bp    length_cm
IND1   IND2   1             1000000     2500000   15.2
IND1   IND3   1             3000000     4200000   12.8
```

### FAM File (PLINK Format)
```
FAM_ID  IND_ID  FATHER  MOTHER  SEX  PHENOTYPE
FAM1    IND1    0       0       1    -9
FAM1    IND2    0       0       2    -9
FAM2    IND3    0       0       1    -9
```

### Ages File (Optional)
```
individual_id    age
IND1            45
IND2            67
IND3            32
```

### Genetic Map File (Optional, for HapIBD)
```
chromosome    position_bp    position_cm
1             1000000        0.5
1             2000000        1.2
1             3000000        1.8
```

### Priors File (Optional)
The priors file allows you to specify age-based relationship constraints. The format is a tab-separated or space-separated file with three columns:

```
rel    operator    age_gap
MHS    >           25
GP     <=          30
```

**Column Descriptions:**
- `rel`: Relationship abbreviation (e.g., MHS for maternal half-siblings, GP for grandparent-grandchild)
- `operator`: Comparison operator (`>`, `<`, `=`, `>=`, `<=`)  
- `age_gap`: Age difference threshold in years

**Example Usage:**
In this example, if two 2nd degree individuals have a >25 year age gap, P(MHS) would be set to 0 and the other probabilities rescaled.

## Examples

### Basic Analysis

Analyze relationships using PhaseIBD output with default parameters:

```bash
python -m ponderosa.cli \
  --ibd segments.txt \
  --map input.map \
  --fam individuals.fam \
  --output my_results
```

### Advanced Analysis with Custom Parameters

```bash
python -m ponderosa.cli \
  --ibd segments.txt \
  --map input.map \
  --fam individuals.fam \
  --ibd-caller hapibd \
  --map genetic.map \
  --min-segment-length 5.0 \
  --min-total-ibd 75.0 \
  --min-probability 0.7 \
  --create-plots \
  --verbose \
  --output detailed_analysis
```

### Analysis with Multiple Chromosome Files

When working with per-chromosome IBD files, use a YAML configuration:

```yaml
files:
  ibd_files:
    - "ibd_chr1.txt"
    - "ibd_chr2.txt" 
    - "ibd_chr3.txt"
    # ... continue for all chromosomes
  fam: "individuals.fam"
  ibd_caller: "hapibd"
  map_files:
    - "genetic_chr1.map"
    - "genetic_chr2.map"
    - "genetic_chr3.map"
    # ... continue for all chromosomes

algorithm:
  min_segment_length: 3.0
  min_total_ibd: 50.0  # Applied after combining all chromosomes

output:
  output: "multi_chr_analysis"
```

```bash
python -m ponderosa.cli --config multi_chromosome_config.yaml
```

### Using Configuration File

```bash
python -m ponderosa.cli --config analysis_config.yaml
```

### Using Configuration File

```bash
python -m ponderosa.cli --config analysis_config.yaml
```

### Analysis with Age Priors

```bash
python -m ponderosa.cli \
  --config base_config.yaml \
  --ages individual_ages.txt \
  --priors age_priors.yaml
```

## Simulation Module

PONDEROSA includes a simulation module for generating training data and testing:

```bash
python -m ponderosa.simulation.cli simulate --config simulation.yaml
```

### Simulation Configuration

The simulation requires a separate YAML configuration:

```yaml
# Required: ped-sim configuration
pedsim:
  pedsim_path: "/path/to/ped-sim"
  vcf_file: "/path/to/founders.vcf"
  random_seed: 12345

# Required: KING output for founder selection  
king_file: "/path/to/kinship.seg"

# Optional: training parameters
training:
  n_pairs_per_relationship: 100
  max_kinship: 0.05

# Optional: processing settings
ibd_caller: "hap-ibd.sh"
output_path: "simulation_output"
cleanup_temp: true
```

## Output Files

PONDEROSA generates several output files with the specified prefix:

- `{prefix}_relationships.txt`: Main results with relationship predictions
- `{prefix}_probabilities.txt`: Detailed probability matrices  
- `{prefix}_pairs.txt`: Processed pair-wise IBD statistics
- `{prefix}_training.pkl`: Trained classifiers (if `--write_training` specified)
- `{prefix}_plots/`: Visualization directory (if `--create-plots` specified)

## Relationship Categories

PONDEROSA classifies relationships into hierarchical categories:

- **1st Degree**: Parent-Child (PC), Full Siblings (FS)
- **2nd Degree**: Grandparent-Grandchild (PGP/MGP), Aunt/Uncle-Niece/Nephew (AV), Half-Siblings (PHS/MHS)
- **3rd Degree**
- **Unrelated**: Distant or no biological relationship

## Performance Considerations

- **Single vs Multiple Files**: Single IBD files are more memory-efficient as they allow Polars to optimize filtering during file reading. Multiple IBD files require loading all data into memory before applying genome-wide filters like `--min-total-ibd`.
- **Memory Usage**: Large datasets may require substantial RAM. Consider filtering pairs with `--min-total-ibd` to reduce memory requirements.
- **Processing Time**: Runtime scales quadratically with the number of individuals. Use `--verbose` to monitor progress.
- **Accuracy**: Results improve with higher quality IBD calls and appropriate parameter tuning for your dataset.

## Troubleshooting

### Common Issues

1. **File Not Found Errors**: Ensure all input files exist and paths are correct
2. **Memory Errors**: Increase `--min-total-ibd` to reduce the number of pairs analyzed
3. **No Results**: Lower `--min-probability` threshold or check IBD segment quality
4. **Format Errors**: Verify IBD file format matches the specified `--ibd-caller`

### Debug Mode

Use `--debug` to see full error tracebacks:

```bash
python -m ponderosa.cli --debug --config config.yaml
```

### Verbose Output

Use multiple `-v` flags for detailed progress information:

```bash
python -m ponderosa.cli -vvv --config config.yaml
```

## Citation

If you use PONDEROSA in your research, please cite:

[add citation]

## License

[add license]

## Support

For questions, bug reports, or feature requests, please [contact information or GitHub issues link].
