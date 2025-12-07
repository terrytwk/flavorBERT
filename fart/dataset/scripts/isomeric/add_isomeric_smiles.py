#!/usr/bin/env python3
"""
Script to replace Canonicalized SMILES with isomeric SMILES from PubChem.

For each entry in the split files, this script:
1. Matches by Canonicalized SMILES to fart_uncurated_with_ids.csv
2. Uses PubChemID, InChI, or INCHIKEY to fetch isomeric SMILES from PubChem
3. Replaces the "Canonicalized SMILES" column with isomeric SMILES where available
4. Removes the "is_multiclass" column
5. Saves output files with the same names (fart_train.csv, fart_val.csv, fart_test.csv) in isomeric-splits folder
6. Skips entries where none of the three identifiers exist (keeps original Canonicalized SMILES)

Performance Optimizations:
- Uses batch processing to fetch up to 100 compounds per API request (much faster)
- Complies with PubChem rate limits: 5 requests/second, 400 requests/minute
- Caches results to avoid duplicate API calls
- Separates PubChem ID lookups (batch) from InChI/InChIKey lookups (individual)
"""

import pandas as pd
import pubchempy as pcp
from tqdm import tqdm
from pathlib import Path
import sys
import time
from collections import defaultdict

# Set up paths
SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR.parent
SPLITS_DIR = DATASET_DIR / 'splits'
OUTPUT_DIR = DATASET_DIR / 'isomeric-splits'
IDS_FILE = DATASET_DIR / 'fart_uncurated_with_ids.csv'

# Split files to process
SPLIT_FILES = ['fart_train.csv', 'fart_val.csv', 'fart_test.csv']

# API rate limiting (requests per second)
# PubChem API limits: 5 requests/second, 400 requests/minute
API_DELAY = 0.2  # 200ms delay between requests (5 requests/second max)
BATCH_SIZE = 100  # Number of CIDs to fetch per batch request
last_api_call_time = 0

# Global cache to avoid duplicate API calls
smiles_cache = {}


def is_valid_identifier(value):
    """Check if an identifier value is valid (not empty, None, nan, or '*')."""
    if pd.isna(value):
        return False
    value_str = str(value).strip()
    return value_str not in ('', 'None', 'nan', '*', 'NaN', 'N/A', 'n/a')


def rate_limit():
    """Enforce rate limiting for API calls."""
    global last_api_call_time
    current_time = time.time()
    elapsed = current_time - last_api_call_time
    if elapsed < API_DELAY:
        time.sleep(API_DELAY - elapsed)
    last_api_call_time = time.time()


def get_cache_key(pubchem_id=None, inchi=None, inchikey=None):
    """Generate a cache key from identifiers."""
    if is_valid_identifier(pubchem_id):
        return f"cid:{str(pubchem_id).strip()}"
    elif is_valid_identifier(inchi):
        return f"inchi:{str(inchi).strip()}"
    elif is_valid_identifier(inchikey):
        return f"inchikey:{str(inchikey).strip()}"
    return None


def get_isomeric_smiles_from_pubchem(pubchem_id=None, inchi=None, inchikey=None):
    """
    Fetch isomeric SMILES from PubChem using one of the identifiers.
    This function is used for InChI/InChIKey lookups only (PubChem IDs are batch processed).
    Uses caching to avoid duplicate API calls.
    
    Args:
        pubchem_id: PubChem CID (int or str)
        inchi: InChI string
        inchikey: INCHIKEY string
    
    Returns:
        isomeric_smiles: str or None
    """
    # Check cache first
    cache_key = get_cache_key(pubchem_id, inchi, inchikey)
    if cache_key and cache_key in smiles_cache:
        return smiles_cache[cache_key]
    
    # If cache miss and no valid identifier, return None
    if not cache_key:
        return None
    
    # Rate limit API calls
    rate_limit()
    
    isomeric_smiles = None
    
    # Try to get CID first (for direct property lookup)
    cid = None
    
    # Try PubChem ID first (most reliable)
    if is_valid_identifier(pubchem_id):
        try:
            cid = int(str(pubchem_id).strip())
        except (ValueError, TypeError):
            pass
    
    # Try InChI to get CID
    if cid is None and is_valid_identifier(inchi):
        try:
            compounds = pcp.get_compounds(str(inchi).strip(), 'inchi')
            if compounds and hasattr(compounds[0], 'cid'):
                cid = compounds[0].cid
        except Exception:
            pass
    
    # Try INCHIKEY to get CID
    if cid is None and is_valid_identifier(inchikey):
        try:
            compounds = pcp.get_compounds(str(inchikey).strip(), 'inchikey')
            if compounds and hasattr(compounds[0], 'cid'):
                cid = compounds[0].cid
        except Exception:
            pass
    
    # If we have a CID, use get_properties directly (faster)
    if cid is not None:
        try:
            props_result = pcp.get_properties(['IsomericSMILES'], cid, 'cid')
            if props_result and len(props_result) > 0:
                result_dict = props_result[0]
                if 'SMILES' in result_dict and result_dict['SMILES']:
                    isomeric_smiles = result_dict['SMILES']
        except Exception:
            pass
    
    # Fallback: Try parsing from compound record if we have InChI/InChIKey but no CID
    if not isomeric_smiles:
        compound = None
        
        if is_valid_identifier(inchi):
            try:
                compounds = pcp.get_compounds(str(inchi).strip(), 'inchi')
                if compounds:
                    compound = compounds[0]
            except Exception:
                pass
        
        if compound is None and is_valid_identifier(inchikey):
            try:
                compounds = pcp.get_compounds(str(inchikey).strip(), 'inchikey')
                if compounds:
                    compound = compounds[0]
            except Exception:
                pass
        
        if compound:
            try:
                if hasattr(compound, 'record') and compound.record:
                    props = compound.record.get('props', [])
                    for prop in props:
                        urn = prop.get('urn', {})
                        label = urn.get('label', '')
                        name = urn.get('name', '')
                        if label == 'SMILES' and name in ('Absolute', 'Isomeric'):
                            value = prop.get('value', {})
                            sval = value.get('sval', '')
                            if sval:
                                isomeric_smiles = sval
                                break
            except Exception:
                pass
    
    # Cache the result (even if None, to avoid retrying failed lookups)
    if cache_key:
        smiles_cache[cache_key] = isomeric_smiles
    
    return isomeric_smiles


def process_split_file(split_file_path, ids_df):
    """
    Process a single split file and replace Canonicalized SMILES with isomeric SMILES.
    Optimized to batch unique identifiers and use caching.
    
    This function:
    - Replaces 'Canonicalized SMILES' column values with isomeric SMILES where available
    - Keeps original Canonicalized SMILES for rows where isomeric SMILES cannot be fetched
    
    Args:
        split_file_path: Path to the split CSV file
        ids_df: DataFrame with identifiers from fart_uncurated_with_ids.csv
    
    Returns:
        DataFrame with Canonicalized SMILES replaced by isomeric SMILES
    """
    print(f"\nProcessing {split_file_path.name}...")
    
    # Read the split file
    split_df = pd.read_csv(split_file_path)
    
    # Drop any unnamed index columns if they exist (first column might be empty from leading comma)
    # This handles the case where the CSV has an index column
    split_df = split_df.loc[:, ~split_df.columns.str.contains('^Unnamed', regex=True)]
    # Also drop empty column name if it exists
    if len(split_df.columns) > 0 and (split_df.columns[0] == '' or split_df.columns[0].startswith('Unnamed')):
        split_df = split_df.drop(columns=[split_df.columns[0]])
    
    # Create a lookup dictionary from ids_df by Canonicalized SMILES
    # Handle multiple matches by taking the first one with identifiers
    ids_lookup = {}
    for _, row in ids_df.iterrows():
        smiles = row['Canonicalized SMILES']
        if smiles not in ids_lookup:
            ids_lookup[smiles] = row
        else:
            # Prefer entries with PubChemID over those without
            current = ids_lookup[smiles]
            if not is_valid_identifier(current['PubChemID']):
                if is_valid_identifier(row['PubChemID']):
                    ids_lookup[smiles] = row
    
    # We'll replace Canonicalized SMILES with isomeric SMILES where available
    # Keep original for rows where we can't fetch isomeric SMILES
    
    # Collect all unique identifiers first to batch process
    unique_identifiers = defaultdict(dict)  # {cache_key: {pubchem_id, inchi, inchikey}}
    identifier_to_rows = defaultdict(list)  # {cache_key: [row_indices]}
    
    print("  Collecting unique identifiers...")
    for idx, row in split_df.iterrows():
        canonical_smiles = row['Canonicalized SMILES']
        
        if canonical_smiles in ids_lookup:
            ids_row = ids_lookup[canonical_smiles]
            pubchem_id = ids_row['PubChemID']
            inchi = ids_row['InChI']
            inchikey = ids_row['INCHIKEY']
            
            has_pubchem = is_valid_identifier(pubchem_id)
            has_inchi = is_valid_identifier(inchi)
            has_inchikey = is_valid_identifier(inchikey)
            
            if has_pubchem or has_inchi or has_inchikey:
                cache_key = get_cache_key(pubchem_id, inchi, inchikey)
                if cache_key:
                    unique_identifiers[cache_key] = {
                        'pubchem_id': pubchem_id if has_pubchem else None,
                        'inchi': inchi if has_inchi else None,
                        'inchikey': inchikey if has_inchikey else None
                    }
                    identifier_to_rows[cache_key].append(idx)
    
    # Fetch isomeric SMILES for all unique identifiers using batch processing
    print(f"  Fetching isomeric SMILES for {len(unique_identifiers)} unique compounds...")
    fetched_results = {}
    
    # Separate compounds by identifier type for batch processing
    # Group PubChem IDs for batch fetching
    pubchem_ids_to_fetch = {}  # {cid: cache_key}
    other_identifiers = {}  # {cache_key: identifiers} for InChI/InChIKey lookups
    
    for cache_key, identifiers in unique_identifiers.items():
        # Check cache first
        if cache_key in smiles_cache:
            fetched_results[cache_key] = smiles_cache[cache_key]
        else:
            # Group by identifier type
            if identifiers['pubchem_id'] and is_valid_identifier(identifiers['pubchem_id']):
                try:
                    cid = int(str(identifiers['pubchem_id']).strip())
                    pubchem_ids_to_fetch[cid] = cache_key
                except (ValueError, TypeError):
                    other_identifiers[cache_key] = identifiers
            else:
                other_identifiers[cache_key] = identifiers
    
    # Batch fetch PubChem IDs
    if pubchem_ids_to_fetch:
        print(f"  Batch fetching {len(pubchem_ids_to_fetch)} compounds by PubChem ID...")
        cids = list(pubchem_ids_to_fetch.keys())
        cache_keys = list(pubchem_ids_to_fetch.values())
        
        # Process in batches
        for i in tqdm(range(0, len(cids), BATCH_SIZE), desc="  Processing batches"):
            batch_cids = cids[i:i+BATCH_SIZE]
            batch_cache_keys = cache_keys[i:i+BATCH_SIZE]
            
            # Rate limit between batches
            rate_limit()
            
            try:
                # Batch fetch isomeric SMILES for this batch of CIDs
                props_result = pcp.get_properties(['IsomericSMILES'], batch_cids, 'cid')
                
                # Track which CIDs we got results for
                result_cids = set()
                
                # Map results back to cache keys
                if props_result:
                    for result in props_result:
                        cid = result.get('CID')
                        isomeric_smiles = result.get('SMILES')
                        
                        if cid and cid in pubchem_ids_to_fetch:
                            cache_key = pubchem_ids_to_fetch[cid]
                            result_cids.add(cid)
                            # Store result (even if None)
                            fetched_results[cache_key] = isomeric_smiles if isomeric_smiles else None
                            smiles_cache[cache_key] = isomeric_smiles if isomeric_smiles else None
                
                # Mark CIDs that didn't return results as None (they might not exist in PubChem)
                for cid in batch_cids:
                    if cid not in result_cids and cid in pubchem_ids_to_fetch:
                        cache_key = pubchem_ids_to_fetch[cid]
                        fetched_results[cache_key] = None
                        # Don't cache None for missing CIDs to allow retry later
                        
            except Exception as e:
                # If batch fails, mark all in batch as failed
                print(f"    Warning: Batch fetch failed: {e}")
                for cid in batch_cids:
                    if cid in pubchem_ids_to_fetch:
                        cache_key = pubchem_ids_to_fetch[cid]
                        fetched_results[cache_key] = None
    
    # Handle remaining identifiers that don't have PubChem IDs (InChI/InChIKey)
    if other_identifiers:
        print(f"  Fetching {len(other_identifiers)} compounds by InChI/InChIKey (individual requests)...")
        for cache_key, identifiers in tqdm(other_identifiers.items(), desc="  Fetching remaining"):
            isomeric_smiles = get_isomeric_smiles_from_pubchem(
                pubchem_id=identifiers['pubchem_id'],
                inchi=identifiers['inchi'],
                inchikey=identifiers['inchikey']
            )
            fetched_results[cache_key] = isomeric_smiles
    
    # Apply results to dataframe - replace Canonicalized SMILES with isomeric SMILES
    print("  Applying results to dataframe...")
    found_identifiers = 0
    fetched_smiles = 0
    fetch_failed = 0
    no_identifiers = 0
    
    for cache_key, row_indices in identifier_to_rows.items():
        isomeric_smiles = fetched_results.get(cache_key)
        found_identifiers += len(row_indices)
        if isomeric_smiles:
            fetched_smiles += len(row_indices)
            for idx in row_indices:
                # Replace Canonicalized SMILES with isomeric SMILES
                split_df.at[idx, 'Canonicalized SMILES'] = isomeric_smiles
        else:
            fetch_failed += len(row_indices)
    
    # Count rows that weren't matched (no identifiers found)
    for idx, row in split_df.iterrows():
        canonical_smiles = row['Canonicalized SMILES']
        if canonical_smiles not in ids_lookup:
            no_identifiers += 1
    
    skipped = no_identifiers + fetch_failed
    
    print(f"  Found identifiers: {found_identifiers}/{len(split_df)} ({found_identifiers/len(split_df)*100:.1f}%)")
    print(f"  Successfully fetched: {fetched_smiles}/{len(split_df)} ({fetched_smiles/len(split_df)*100:.1f}%)")
    print(f"  Fetch failed: {fetch_failed}/{len(split_df)} ({fetch_failed/len(split_df)*100:.1f}%)")
    print(f"  No identifiers: {no_identifiers}/{len(split_df)} ({no_identifiers/len(split_df)*100:.1f}%)")
    print(f"  Cache hits: {len([k for k in unique_identifiers.keys() if k in smiles_cache])}/{len(unique_identifiers)}")
    
    return split_df


def main():
    """Main function to process all split files."""
    print("=" * 60)
    print("Replacing Canonicalized SMILES with Isomeric SMILES")
    print("=" * 60)
    
    # Check if files exist
    if not IDS_FILE.exists():
        print(f"Error: {IDS_FILE} not found!")
        sys.exit(1)
    
    if not SPLITS_DIR.exists():
        print(f"Error: {SPLITS_DIR} not found!")
        sys.exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load the identifiers file
    print(f"\nLoading identifiers from {IDS_FILE.name}...")
    ids_df = pd.read_csv(IDS_FILE)
    print(f"Loaded {len(ids_df):,} entries with identifiers")
    
    # Process each split file
    for split_filename in SPLIT_FILES:
        split_file_path = SPLITS_DIR / split_filename
        
        if not split_file_path.exists():
            print(f"Warning: {split_file_path} not found, skipping...")
            continue
        
        # Process the split file (replaces Canonicalized SMILES with isomeric SMILES)
        result_df = process_split_file(split_file_path, ids_df)
        
        # Remove is_multiclass column if it exists
        if 'is_multiclass' in result_df.columns:
            result_df = result_df.drop(columns=['is_multiclass'])
            print(f"  Removed 'is_multiclass' column")
        
        # Save the result with the same filename in the output directory
        output_path = OUTPUT_DIR / split_filename
        result_df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

