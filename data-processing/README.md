# Data Processing - fetch-data Script

## Overview

The `fetch-data.py` script downloads and extracts compound SMILES strings from PubChem and FoodDB databases.

## Usage

```bash
cd data-processing/script
python fetch-data.py
```

When prompted, select a data source:
- **Option 1 (PubChem)**: Downloads all compound data from PubChem FTP server and extracts isomeric SMILES from SDF files
- **Option 2 (FoodDB)**: Downloads FoodDB JSON zip file (or uses existing file if present) and extracts SMILES from JSON format

## Output

- **PubChem**: `data-processing/data/pubchem/pubchem_isomeric_smiles.txt`
- **FoodDB**: `data-processing/data/foodb/foodb_smiles.txt`

Each line contains a single SMILES string. The script will prompt before overwriting existing output files.

## Notes

- FoodDB zip files are cached in `data-processing/data/foodb/` and reused if already downloaded
- Temporary files are automatically cleaned up on exit
- Processing may take several hours for large datasets (PubChem)
