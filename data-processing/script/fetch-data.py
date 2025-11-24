#!/usr/bin/env python3
"""
Script to fetch compound data from various sources.
Currently supports PubChem database via FTP bulk download and FoodDB.
"""

import gzip
import requests
from pathlib import Path
from tqdm import tqdm
import tempfile
import re
import zipfile
import json
import shutil

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Will parse SDF manually (slower).")


def parse_sdf_manual(sdf_file):
    """
    Manually parse SDF file to extract PUBCHEM_SMILES without RDKit.
    
    Args:
        sdf_file (str): Path to SDF file
        
    Yields:
        str: SMILES string
    """
    current_compound_text = []
    
    with open(sdf_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_stripped = line.strip()
            current_compound_text.append(line_stripped)
            
            # When we hit $$$$, we've reached the end of a compound
            if line_stripped == '$$$$':
                # Look for PUBCHEM_SMILES in this compound
                compound_text = '\n'.join(current_compound_text)
                
                # Try different property names
                for prop_name in ['PUBCHEM_SMILES', 'PUBCHEM_OPENEYE_ISOMERIC_SMILES',
                                 'PUBCHEM_OPENEYE_CANONICAL_SMILES']:
                    prop_marker = f'> <{prop_name}>'
                    if prop_marker in compound_text:
                        # Find the value after the property marker
                        idx = compound_text.find(prop_marker)
                        if idx != -1:
                            # Get text after the marker until next property or end
                            after_marker = compound_text[idx + len(prop_marker):]
                            # Find the next property marker or $$$$
                            next_prop = after_marker.find('> <')
                            if next_prop != -1:
                                after_marker = after_marker[:next_prop]
                            else:
                                # Remove trailing $$$$
                                after_marker = after_marker.replace('$$$$', '').strip()
                            
                            # Get first non-empty line
                            for subline in after_marker.split('\n'):
                                subline = subline.strip()
                                if subline and not subline.startswith('>'):
                                    yield subline
                                    break
                        # Found a SMILES, move to next compound
                        break
                
                # Reset for next compound (keep only last 10 lines to track position)
                current_compound_text = current_compound_text[-10:] if len(current_compound_text) > 10 else []


def extract_smiles_from_sdf(sdf_file, output_file, use_rdkit=True):
    """
    Extract SMILES strings from an SDF file and append to output file.
    
    Args:
        sdf_file (str): Path to SDF file
        output_file (str): Path to output file to append SMILES
        use_rdkit (bool): Whether to use RDKit for parsing
        
    Returns:
        int: Number of SMILES extracted
    """
    count = 0
    
    if use_rdkit and RDKIT_AVAILABLE:
        # Use RDKit for parsing (faster and more reliable)
        suppl = Chem.SDMolSupplier(sdf_file, sanitize=False)
        
        with open(output_file, 'a') as f_out:
            for mol in suppl:
                if mol is not None:
                    # Try different property names for SMILES
                    smiles = None
                    for prop_name in ['PUBCHEM_SMILES', 'PUBCHEM_OPENEYE_ISOMERIC_SMILES', 
                                     'PUBCHEM_OPENEYE_CANONICAL_SMILES', 'IsomericSMILES']:
                        try:
                            if mol.HasProp(prop_name):
                                smiles = mol.GetProp(prop_name)
                                break
                        except:
                            continue
                    
                    if smiles and smiles.strip():
                        f_out.write(f"{smiles.strip()}\n")
                        count += 1
    else:
        # Manual parsing
        with open(output_file, 'a') as f_out:
            for smiles in parse_sdf_manual(sdf_file):
                if smiles:
                    f_out.write(f"{smiles}\n")
                    count += 1
    
    return count


def get_sdf_file_list(ftp_base="https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/"):
    """
    Get list of all SDF files from PubChem FTP server.
    
    Args:
        ftp_base (str): Base URL for PubChem FTP server
        
    Returns:
        list: List of SDF filenames
    """
    print("Fetching list of SDF files from PubChem FTP server...")
    response = requests.get(ftp_base)
    response.raise_for_status()
    
    # Parse HTML directory listing using regex
    # Look for links to .sdf.gz files
    pattern = r'href="(Compound_\d+_\d+\.sdf\.gz)"'
    sdf_files = re.findall(pattern, response.text)
    
    # Sort files by compound ID range
    sdf_files.sort()
    
    print(f"Found {len(sdf_files)} SDF files")
    return sdf_files


def download_and_process_sdf(sdf_filename, ftp_base, output_file, temp_dir):
    """
    Download, decompress, and process a single SDF file.
    
    Args:
        sdf_filename (str): Name of the SDF file to download
        ftp_base (str): Base URL for PubChem FTP server
        output_file (str): Path to output file
        temp_dir (Path): Temporary directory for downloads
        
    Returns:
        int: Number of SMILES extracted
    """
    sdf_url = ftp_base + sdf_filename
    
    compressed_file = temp_dir / sdf_filename
    sdf_file = temp_dir / sdf_filename.replace('.gz', '')
    
    # Download the gzipped SDF file
    response = requests.get(sdf_url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    with open(compressed_file, 'wb') as f, tqdm(
        desc=f"Downloading {sdf_filename}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        leave=False
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    # Decompress the file
    with gzip.open(compressed_file, 'rb') as f_in:
        with open(sdf_file, 'wb') as f_out:
            with tqdm(desc=f"Decompressing {sdf_filename}", unit='B', unit_scale=True, leave=False) as pbar:
                while True:
                    chunk = f_in.read(8192)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    pbar.update(len(chunk))
    
    # Extract SMILES from the SDF file
    count = extract_smiles_from_sdf(str(sdf_file), output_file, use_rdkit=RDKIT_AVAILABLE)
    
    # Clean up temporary files
    compressed_file.unlink()
    sdf_file.unlink()
    
    return count


def fetch_pubchem_smiles(output_file="data-processing/data/pubchem/pubchem_isomeric_smiles.txt", 
                         ftp_base="https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/"):
    """
    Download all PubChem SDF files from FTP and extract SMILES strings.
    
    Args:
        output_file (str): Output file path for SMILES strings
        ftp_base (str): Base URL for PubChem FTP server
    """
    print("=" * 60)
    print("PubChem Compound Data Fetcher")
    print("=" * 60)
    print("Note: This uses PUBCHEM_SMILES (isomeric SMILES with stereochemistry)")
    print(f"FTP Server: {ftp_base}")
    print("\nThis will download and process ALL PubChem compounds.")
    print("This may take several hours depending on your connection and processing speed.")
    
    # Get list of all SDF files
    sdf_files = get_sdf_file_list(ftp_base)
    
    if not sdf_files:
        print("Error: No SDF files found on FTP server.")
        return
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear output file if it exists (start fresh)
    if output_path.exists():
        response = input(f"\nOutput file {output_path} already exists. Overwrite? (y/n): ").strip().lower()
        if response == 'y':
            output_path.unlink()
        else:
            print("Appending to existing file...")
    
    # Create temporary directory for downloads
    temp_dir = Path(tempfile.mkdtemp())
    
    # Store temp directory path for potential manual cleanup
    print(f"Temporary directory: {temp_dir}")
    print("(This will be automatically cleaned up on exit)")
    
    total_smiles = 0
    
    try:
        # Process each SDF file
        for i, sdf_filename in enumerate(tqdm(sdf_files, desc="Processing files"), 1):
            print(f"\n[{i}/{len(sdf_files)}] Processing {sdf_filename}...")
            
            try:
                count = download_and_process_sdf(sdf_filename, ftp_base, str(output_path), temp_dir)
                total_smiles += count
                print(f"  Extracted {count} SMILES from {sdf_filename}")
            except Exception as e:
                print(f"  Error processing {sdf_filename}: {e}")
                continue
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up temporary files...")
        raise
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        print(f"Temporary files may remain at: {temp_dir}")
        raise
    finally:
        # Clean up temporary directory
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                print(f"\nCleaned up temporary directory: {temp_dir}")
        except Exception as cleanup_error:
            print(f"\nWarning: Could not clean up temporary directory: {cleanup_error}")
            print(f"Please manually remove: {temp_dir}")
    
    print(f"\n{'='*60}")
    print(f"Successfully extracted {total_smiles} SMILES strings")
    print(f"SMILES strings saved to: {output_path.absolute()}")
    print(f"{'='*60}")


def fetch_foodb_smiles(output_file="data-processing/data/foodb/foodb_smiles.txt",
                       zip_url="https://foodb.ca/public/system/downloads/foodb_2020_04_07_json.zip"):
    """
    Download FoodDB JSON zip file and extract SMILES strings.
    
    Args:
        output_file (str): Output file path for SMILES strings
        zip_url (str): URL to download FoodDB JSON zip file
    """
    print("=" * 60)
    print("FoodDB Compound Data Fetcher")
    print("=" * 60)
    print(f"Download URL: {zip_url}")
    print("\nThis will download and process FoodDB compounds.")
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear output file if it exists (start fresh)
    if output_path.exists():
        response = input(f"\nOutput file {output_path} already exists. Overwrite? (y/n): ").strip().lower()
        if response == 'y':
            output_path.unlink()
        else:
            print("Appending to existing file...")
    
    # Check if zip file already exists in data directory
    data_dir = output_path.parent
    zip_filename = Path(zip_url).name  # Extract filename from URL
    local_zip_file = data_dir / zip_filename
    
    # Create temporary directory for extraction
    temp_dir = Path(tempfile.mkdtemp())
    extract_dir = temp_dir / "foodb_extracted"
    extract_dir.mkdir(exist_ok=True)
    
    # Store temp directory path for potential manual cleanup
    print(f"Temporary directory: {temp_dir}")
    print("(This will be automatically cleaned up on exit)")
    
    try:
        # Check if zip file exists locally
        if local_zip_file.exists():
            print(f"\nFound existing zip file: {local_zip_file}")
            print("Using existing file instead of downloading.")
            zip_file = local_zip_file
        else:
            # Download the zip file
            print("\nDownloading FoodDB JSON zip file...")
            zip_file = temp_dir / zip_filename
            
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(zip_file, 'wb') as f, tqdm(
                desc=f"Downloading {zip_filename}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Save downloaded file to data directory for future use
            print(f"\nSaving zip file to {local_zip_file} for future use...")
            shutil.copy2(zip_file, local_zip_file)
            print("Zip file saved successfully.")
        
        # Extract the zip file
        print("\nExtracting zip file...")
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find Compound.json file
        compound_json = extract_dir / "Compound.json"
        
        # Also check if it's in a subdirectory (common with zip files)
        if not compound_json.exists():
            # Search for Compound.json in extracted directory
            found_files = list(extract_dir.rglob("Compound.json"))
            if found_files:
                compound_json = found_files[0]
            else:
                raise FileNotFoundError("Could not find Compound.json in extracted zip file")
        
        print(f"\nFound Compound.json at: {compound_json}")
        print("Extracting SMILES strings...")
        
        # Count total lines for progress bar
        print("Counting lines in Compound.json...")
        with open(compound_json, 'r', encoding='utf-8') as f_in:
            total_lines = sum(1 for line in f_in if line.strip())
        
        # Parse JSONL (JSON Lines) format - each line is a separate JSON object
        count = 0
        with open(compound_json, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w') as f_out:
                # Read file line by line for JSONL format
                for line in tqdm(f_in, desc="Processing compounds", total=total_lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        compound = json.loads(line)
                        # Extract moldb_smiles if it exists
                        if isinstance(compound, dict) and 'moldb_smiles' in compound:
                            smiles = compound['moldb_smiles']
                            if smiles and isinstance(smiles, str) and smiles.strip():
                                f_out.write(f"{smiles.strip()}\n")
                                count += 1
                    except json.JSONDecodeError as e:
                        # Silently skip malformed lines
                        continue
        
        print(f"\n{'='*60}")
        print(f"Successfully extracted {count} SMILES strings")
        print(f"SMILES strings saved to: {output_path.absolute()}")
        print(f"{'='*60}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up temporary files...")
        raise
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        print(f"Temporary files may remain at: {temp_dir}")
        raise
    finally:
        # Clean up temporary directory
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                print(f"\nCleaned up temporary directory: {temp_dir}")
        except Exception as cleanup_error:
            print(f"\nWarning: Could not clean up temporary directory: {cleanup_error}")
            print(f"Please manually remove: {temp_dir}")


def main():
    """Main function to prompt user and fetch data."""
    print("=" * 60)
    print("Data Fetching Script")
    print("=" * 60)
    print("\nAvailable data sources:")
    print("  (1) pubchem - PubChem compound database")
    print("  (2) foodb - FoodDB compound database")
    
    choice = input("\nSelect data source (enter number): ").strip()
    
    if choice == '1':
        output_file = "data-processing/data/pubchem/pubchem_isomeric_smiles.txt"
        fetch_pubchem_smiles(output_file=output_file)
    elif choice == '2':
        output_file = "data-processing/data/foodb/foodb_smiles.txt"
        fetch_foodb_smiles(output_file=output_file)
    else:
        print(f"\nError: '{choice}' is not a valid option.")
        print("Please enter '1' to fetch PubChem data or '2' to fetch FoodDB data.")


if __name__ == "__main__":
    main()
