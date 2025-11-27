# Data Processing Scripts

## Overview

This directory contains scripts for downloading and processing compound data from various sources.

## Scripts

### 1. fetch-data.py

Downloads and extracts compound SMILES strings from PubChem and FoodDB databases.

#### Usage

```bash
cd data-processing/script
python fetch-data.py
```

When prompted, select a data source:
- **Option 1 (PubChem)**: Downloads all compound data from PubChem FTP server and extracts isomeric SMILES from SDF files
- **Option 2 (FoodDB)**: Downloads FoodDB JSON zip file (or uses existing file if present) and extracts SMILES from JSON format

#### Output

- **PubChem**: `data-processing/data/pubchem/pubchem_isomeric_smiles.txt`
- **FoodDB**: `data-processing/data/foodb/foodb_smiles.txt`

Each line contains a single SMILES string. The script will prompt before overwriting existing output files.

#### Notes

- FoodDB zip files are cached in `data-processing/data/foodb/` and reused if already downloaded
- Temporary files are automatically cleaned up on exit
- Processing may take several hours for large datasets (PubChem)

---

### 2. extract-random-smiles.py

Extracts random compounds from FoodDB's `Compound_cleaned.csv` file and writes canonicalized, non-isomeric SMILES strings to an output file. Optionally supports SMILES augmentation to generate multiple random variations of each compound.

#### Usage

```bash
cd data-processing/script
python extract-random-smiles.py [OPTIONS]
```

#### Command-line Options

- `-n, --num-compounds N`: Number of random compounds to extract (if not provided, will prompt for input)
- `-i, --input PATH`: Input CSV file path (default: `data/foodb/Compound_cleaned.csv`)
- `-o, --output PATH`: Output text file path (default: `data/processed/foodb_<N>_smiles.txt` or `foodb_<N>_augmented_smiles.txt` if augmentation is enabled)
- `--augment`: Enable SMILES augmentation (generates 10 random variations per original compound)

#### Examples

```bash
# Extract 1000 random compounds (interactive prompts)
python extract-random-smiles.py

# Extract 5000 compounds with explicit arguments
python extract-random-smiles.py -n 5000

# Extract 1000 compounds with augmentation (10 variations each = 10,000 total SMILES)
python extract-random-smiles.py -n 1000 --augment

# Custom input and output paths
python extract-random-smiles.py -n 2000 -i /path/to/Compound_cleaned.csv -o /path/to/output.txt
```

#### Output

- **Without augmentation**: `data-processing/data/processed/foodb_<N>_smiles.txt`
- **With augmentation**: `data-processing/data/processed/foodb_<N>_augmented_smiles.txt`

Each line contains a single canonicalized, non-isomeric SMILES string (stereochemistry removed).

#### Features

- **Canonicalization**: All SMILES are canonicalized using RDKit (non-isomeric, no stereochemistry)
- **Random Sampling**: Compounds are randomly sampled from all available compounds with valid SMILES
- **SMILES Augmentation**: Optional generation of multiple random SMILES representations per compound (useful for data augmentation in ML models)
- **Validation**: Only valid SMILES are included in the output
- **Progress Tracking**: Shows progress during reading, sampling, canonicalization, and augmentation steps

#### Notes

- Requires RDKit to be installed
- The script reads the `moldb_smiles` column from the CSV file
- Augmentation uses the maxsmi library approach to generate unique random SMILES variations
- If augmentation fails for a compound, the original canonicalized SMILES is used
