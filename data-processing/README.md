# Data Processing - fetch-data Script

## Overview

The `fetch-data.py` script is a tool for downloading and extracting compound data from various chemical databases. Currently, it supports fetching compound data from the PubChem database via FTP bulk download and extracting SMILES (Simplified Molecular Input Line Entry System) strings from SDF (Structure Data Format) files.

## Features

- **PubChem Database Support**: Downloads all compound data from PubChem's FTP server
- **SMILES Extraction**: Extracts isomeric SMILES strings (with stereochemistry) from SDF files
- **Dual Parsing Modes**: 
  - Uses RDKit for fast and reliable parsing (if available)
  - Falls back to manual SDF parsing if RDKit is not installed
- **Progress Tracking**: Shows download and processing progress with progress bars
- **Error Handling**: Continues processing even if individual files fail
- **Temporary File Management**: Automatically cleans up downloaded files after processing

## Requirements

### Required Dependencies

- Python 3.6+
- `requests` - For downloading files from FTP server
- `tqdm` - For progress bars

### Optional Dependencies

- `rdkit` - For faster and more reliable SDF parsing (highly recommended)

### Installation

```bash
# Install required packages
pip install requests tqdm

# Optional: Install RDKit for better performance
# Note: RDKit installation can be complex, see https://www.rdkit.org/docs/Install.html
conda install -c conda-forge rdkit
# or
pip install rdkit-pypi  # (if available for your Python version)
```

## Usage

### Running the Script

```bash
cd data-processing/script
python fetch-data.py
```

The script will:
1. Display available data sources
2. Prompt you to select a data source (currently only PubChem is available)
3. Download and process all compound data from the selected source

### Command Line Options

The script uses an interactive menu. When prompted:
- Enter `1` to fetch PubChem data

### Output

The script saves extracted SMILES strings to:
```
data-processing/data/pubchem/pubchem_isomeric_smiles.txt
```

Each line in the output file contains a single SMILES string:
```
CCO
CC(C)O
C1=CC=CC=C1
...
```

## How It Works

### 1. File Discovery
The script first fetches the list of available SDF files from the PubChem FTP server by parsing the HTML directory listing.

### 2. Download Process
For each SDF file:
- Downloads the gzipped SDF file from the FTP server
- Decompresses the file
- Extracts SMILES strings
- Cleans up temporary files

### 3. SMILES Extraction

The script attempts to extract SMILES in the following priority order:
1. `PUBCHEM_SMILES` - Primary isomeric SMILES with stereochemistry
2. `PUBCHEM_OPENEYE_ISOMERIC_SMILES` - Alternative isomeric SMILES
3. `PUBCHEM_OPENEYE_CANONICAL_SMILES` - Canonical SMILES (without stereochemistry)
4. `IsomericSMILES` - Generic isomeric SMILES property

### 4. Parsing Methods

**With RDKit (recommended):**
- Uses `Chem.SDMolSupplier` to parse SDF files
- Faster and more reliable
- Better error handling

**Without RDKit (fallback):**
- Manual parsing of SDF file format
- Looks for property markers (`> <PROPERTY_NAME>`)
- Extracts values between property markers and compound boundaries (`$$$$`)

## Performance Considerations

- **Processing Time**: Downloading and processing all PubChem compounds can take several hours depending on:
  - Internet connection speed
  - Processing power
  - Whether RDKit is available (significantly faster)
  
- **Storage**: The script uses temporary storage for downloaded files, which are automatically cleaned up after processing.

- **Memory**: The script processes files one at a time to minimize memory usage.

## Error Handling

- If a specific SDF file fails to download or process, the script logs the error and continues with the next file
- Network errors are handled gracefully
- The script will prompt before overwriting an existing output file

## Output File Management

- If the output file already exists, the script will ask whether to overwrite it or append to it
- The output directory is created automatically if it doesn't exist

## Technical Details

### SDF File Format
SDF (Structure Data Format) files contain:
- Molecular structure data
- Properties in the format: `> <PROPERTY_NAME>` followed by the value
- Compound boundaries marked with `$$$$`

### FTP Server
The script connects to:
```
https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/
```

This contains all current PubChem compound data in SDF format, organized into multiple files (e.g., `Compound_000000001_000500000.sdf.gz`).

## Future Enhancements

The script is designed to be extensible. Additional data sources can be added by:
1. Implementing a new fetch function (similar to `fetch_pubchem_smiles`)
2. Adding the option to the interactive menu in `main()`

## Troubleshooting

### RDKit Installation Issues
If RDKit is not available, the script will automatically fall back to manual parsing. However, this is slower. For better performance, try installing RDKit using conda:
```bash
conda install -c conda-forge rdkit
```

### Network Timeouts
If you experience network timeouts, the script will continue with the next file. You can re-run the script and it will append to the existing output file (if you choose not to overwrite).

### Memory Issues
The script processes one file at a time to minimize memory usage. If you still encounter memory issues, consider processing files in smaller batches.

## License

This script is part of the flavorBERT project.

