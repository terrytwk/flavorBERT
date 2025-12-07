#!/usr/bin/env python3
"""
FART Data Extraction Script

This script extracts data from individual datasets and combines them into
fart_uncurated_with_ids.csv with additional PubChemID, InChI, and INCHIKEY columns.
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from rdkit import Chem
from tqdm import tqdm
import os
from pathlib import Path

# Set up paths
SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR.parent / 'individual-datasets'
OUTPUT_FILE = SCRIPT_DIR.parent / 'fart_uncurated_with_ids.csv'

# Configuration: Set to True to enable PubChem API calls for missing InChI/INCHIKEY
# Default is False - will only use data already present in the datasets
USE_PUBCHEM_API = False

# Optional import for PubChem API (only needed if USE_PUBCHEM_API is True)
PUBCHEMPY_AVAILABLE = False
if USE_PUBCHEM_API:
    try:
        import pubchempy as pcp
        PUBCHEMPY_AVAILABLE = True
    except ImportError:
        print("Warning: pubchempy not installed. Set USE_PUBCHEM_API = False or install with: pip install pubchempy")

# Initialize the final dataframe with additional columns
columns = ['Canonicalized SMILES', 'Canonicalized Taste', 'Original Labels', 'Source', 
           'PubChemID', 'InChI', 'INCHIKEY']
fart_uncurated = pd.DataFrame(columns=columns)

# Arrays for tracking dataset size
full_dataset_size = [0]
dataset_stage = ['initialization']


# ============================================================================
# Utility Functions
# ============================================================================

def split_flavors(flavor_string):
    """Split the flavors from a string 'flavor1, flavor2, ...' into a list of flavors."""
    if not isinstance(flavor_string, str):
        return []
    return flavor_string.split(', ')


def flavors_to_list(data, flavor_column_name='Flavor'):
    """Transform the row 'flavor_column_name' so that instead of one string value it
    contains the list of all strings."""
    data[flavor_column_name] = data[flavor_column_name].apply(lambda x: split_flavors(x))
    return data


def distinct_flavors(data, flavor_column_name="Flavor"):
    """Return dictionary {'flavor': number_of_occurrences}."""
    data = flavors_to_list(data.copy(), flavor_column_name)
    all_flavors = [flavor for sublist in data[flavor_column_name] for flavor in sublist]
    taste_counts = Counter(all_flavors)
    taste_counts_dict = dict(taste_counts)
    taste_counts_dict = dict(sorted(taste_counts_dict.items(), key=lambda item: item[1], reverse=True))
    return taste_counts_dict


def tidy_canonical_flavors(df):
    """Flatten the canonical flavors list into individual rows."""
    new_rows = []
    for index, row in df.iterrows():
        if isinstance(row.get('Canonicalized Taste Intermediary'), list):
            flavors = row['Canonicalized Taste Intermediary']
            for flavor in flavors:
                new_row = row.copy()
                new_row['Canonicalized Taste Intermediary'] = flavor
                new_row['Canonicalized Taste'] = str(flavor)
                new_rows.append(new_row)
        else:
            row['Canonicalized Taste'] = str(row.get('Canonicalized Taste Intermediary', ''))
            new_rows.append(row)
    return pd.DataFrame(new_rows)


def canonicalize_smiles(smiles):
    """Canonicalize SMILES strings."""
    if pd.isna(smiles) or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return None


def canonicalize_smiles_column(df, smiles_name='SMILES'):
    """Canonicalize SMILES strings in a DataFrame column."""
    df = df.dropna(subset=[smiles_name])
    df['Canonicalized SMILES'] = df[smiles_name].apply(canonicalize_smiles)
    df = df[df['Canonicalized SMILES'].notna()]
    return df


def get_pubchem_data(cid):
    """Get InChI and INCHIKEY from PubChem CID."""
    if not PUBCHEMPY_AVAILABLE:
        return None, None, None
    
    if pd.isna(cid) or cid == '' or cid is None:
        return None, None, None
    
    try:
        # Convert to int if possible
        if isinstance(cid, str):
            cid = cid.strip()
            if cid == '':
                return None, None, None
            cid = int(cid)
        
        compound = pcp.Compound.from_cid(cid)
        inchi = getattr(compound, 'inchi', None)
        inchikey = getattr(compound, 'inchikey', None)
        return cid, inchi, inchikey
    except (ValueError, TypeError, Exception):
        # Silently fail if CID is invalid or API call fails
        return None, None, None


def extract_identifiers_from_dataset(df, cid_column=None, inchi_column=None, inchikey_column=None):
    """Extract PubChemID, InChI, and INCHIKEY from existing columns in the dataset.
    
    Args:
        df: DataFrame to enrich
        cid_column: Name of column containing PubChem CIDs (e.g., 'PubChem CID', 'PubChem ID')
        inchi_column: Name of column containing InChI strings
        inchikey_column: Name of column containing INCHIKEY strings
    """
    # Initialize columns if they don't exist
    if 'PubChemID' not in df.columns:
        df['PubChemID'] = None
    if 'InChI' not in df.columns:
        df['InChI'] = None
    if 'INCHIKEY' not in df.columns:
        df['INCHIKEY'] = None
    
    # Extract PubChemID from CID column if available
    if cid_column and cid_column in df.columns:
        df['PubChemID'] = df[cid_column].where(df[cid_column].notna() & (df[cid_column] != ''), None)
    
    # Extract InChI from existing column if available
    if inchi_column and inchi_column in df.columns:
        df['InChI'] = df[inchi_column].where(df[inchi_column].notna() & (df[inchi_column] != ''), df['InChI'])
    
    # Extract INCHIKEY from existing column if available
    if inchikey_column and inchikey_column in df.columns:
        df['INCHIKEY'] = df[inchikey_column].where(df[inchikey_column].notna() & (df[inchikey_column] != ''), df['INCHIKEY'])
    
    return df


def enrich_with_pubchem_api(df, cid_column=None):
    """Optionally enrich dataframe with InChI and INCHIKEY from PubChem API.
    
    Only fetches data for rows where InChI or INCHIKEY are missing.
    Requires internet connection.
    """
    if not cid_column or cid_column not in df.columns:
        return df
    
    print(f"Fetching missing InChI and INCHIKEY from PubChem API using {cid_column} column...")
    print("Note: This requires internet connection and makes API calls to PubChem.")
    
    # Only fetch for rows where we have a CID but missing InChI or INCHIKEY
    mask = df[cid_column].notna() & (df[cid_column] != '') & \
           ((df['InChI'].isna()) | (df['INCHIKEY'].isna()))
    
    rows_to_fetch = df[mask]
    if len(rows_to_fetch) == 0:
        print("No missing data to fetch from API.")
        return df
    
    results = []
    for idx, row in tqdm(rows_to_fetch.iterrows(), total=len(rows_to_fetch), desc="Fetching PubChem data"):
        cid = row[cid_column]
        _, inchi, inchikey = get_pubchem_data(cid)
        results.append({
            'idx': idx,
            'InChI': inchi if inchi else row.get('InChI'),
            'INCHIKEY': inchikey if inchikey else row.get('INCHIKEY')
        })
    
    # Update dataframe with fetched data
    for result in results:
        idx = result['idx']
        if result['InChI']:
            df.at[idx, 'InChI'] = result['InChI']
        if result['INCHIKEY']:
            df.at[idx, 'INCHIKEY'] = result['INCHIKEY']
    
    return df


# ============================================================================
# Process ChemTastesDB
# ============================================================================

print("\n=== Processing ChemTastesDB ===")
chemtastes_db = pd.read_csv(DATASET_DIR / 'chemtastes_db.csv')
chemtastes_db = chemtastes_db.dropna(subset=["Class taste"])

FLAVOR_CATEGORIES_map_chemtastes = {
    'Sweetness': 'sweet',
    'Bitterness': 'bitter',
    'Umaminess': 'umami',
    'Sourness': 'sour'
}

FLAVOR_CATEGORIES_map_chemtastes_multiflavors = {
    'Acid': 'sour',
    'Bitter': 'bitter',
    'Moderately bitter': 'bitter',
    'Sour': 'sour',
    'Sweet': 'sweet',
    'Umami': 'umami'
}

def map_flavor(row):
    class_taste = row["Class taste"]
    if class_taste == "Multitaste":
        return 'multi'
    else:
        return [FLAVOR_CATEGORIES_map_chemtastes.get(class_taste, 'undefined')]

chemtastes_db['Canonicalized Taste Intermediary'] = chemtastes_db.apply(map_flavor, axis=1)

# Handle multitaste
def split_flavors_multitaste(text):
    if pd.isna(text):
        return []
    split_list = re.split(r'[;,/]', str(text))
    return [word.strip() for word in split_list if word.strip()]

chemtastes_db.loc[chemtastes_db["Class taste"].apply(lambda x: str(x)[0] == "M" if pd.notna(x) else False), "Taste"] = \
    chemtastes_db.loc[chemtastes_db["Class taste"].apply(lambda x: str(x)[0] == "M" if pd.notna(x) else False), "Taste"].apply(split_flavors_multitaste)

def translate_flavors(item_list):
    if not isinstance(item_list, list):
        return ['undefined']
    return [FLAVOR_CATEGORIES_map_chemtastes_multiflavors.get(item, 'undefined') for item in item_list]

for idx, row in chemtastes_db[chemtastes_db['Canonicalized Taste Intermediary'] == 'multi'].iterrows():
    chemtastes_db.at[idx, "Canonicalized Taste Intermediary"] = translate_flavors(row.get("Taste", []))

chemtastes_db = tidy_canonical_flavors(chemtastes_db)
chemtastes_db = canonicalize_smiles_column(chemtastes_db, 'canonical SMILES')
chemtastes_db['Source'] = 'ChemTastesDB'
chemtastes_db['Original Labels'] = chemtastes_db['Taste'].astype(str) + ', ' + chemtastes_db['Class taste'].astype(str)

# Extract identifiers from existing columns
chemtastes_db = extract_identifiers_from_dataset(chemtastes_db, cid_column='PubChem CID')

# Optionally enrich with API if enabled
if USE_PUBCHEM_API:
    chemtastes_db = enrich_with_pubchem_api(chemtastes_db, cid_column='PubChem CID')

chemtastes_db_subset = chemtastes_db[['Canonicalized SMILES', 'Canonicalized Taste', 'Original Labels', 'Source', 
                                      'PubChemID', 'InChI', 'INCHIKEY']]
chemtastes_db_subset = chemtastes_db_subset.reset_index(drop=True)
fart_uncurated = pd.concat([fart_uncurated, chemtastes_db_subset], axis=0, ignore_index=True)

full_dataset_size.append(len(fart_uncurated))
dataset_stage.append('ChemtastesDB')
print(f"Added {len(chemtastes_db_subset)} molecules from ChemTastesDB")


# ============================================================================
# Process FlavorDB
# ============================================================================

print("\n=== Processing FlavorDB ===")
flavor_db = pd.read_csv(DATASET_DIR / 'flavor_db.csv')
flavor_db = flavor_db.dropna(subset=['Flavor'])

def canonicalize_flavors(flavor_list):
    assigned_flavors = []
    if isinstance(flavor_list, str):
        flavor_list = split_flavors(flavor_list)
    if ('sweet' in flavor_list) or ('sweet-like' in flavor_list):
        assigned_flavors.append('sweet')
    if ('bitter' in flavor_list):
        assigned_flavors.append('bitter')
    if ('sour' in flavor_list) or ('acid' in flavor_list):
        assigned_flavors.append('sour')
    if len(assigned_flavors) == 0:
        assigned_flavors.append('undefined')
    return assigned_flavors

flavor_db['Canonicalized Taste Intermediary'] = flavor_db['Flavor'].apply(lambda x: canonicalize_flavors(x))
flavor_db = tidy_canonical_flavors(flavor_db)
flavor_db = canonicalize_smiles_column(flavor_db, 'SMILES')
flavor_db['Source'] = 'flavor_db'
flavor_db['Original Labels'] = flavor_db['Flavor']

# Extract identifiers from existing columns
flavor_db = extract_identifiers_from_dataset(flavor_db, cid_column='PubChem ID')

# Optionally enrich with API if enabled
if USE_PUBCHEM_API:
    flavor_db = enrich_with_pubchem_api(flavor_db, cid_column='PubChem ID')

flavor_db_subset = flavor_db[['Canonicalized SMILES', 'Canonicalized Taste', 'Original Labels', 'Source',
                              'PubChemID', 'InChI', 'INCHIKEY']]
flavor_db_subset = flavor_db_subset.reset_index(drop=True)
fart_uncurated = pd.concat([fart_uncurated, flavor_db_subset], axis=0, ignore_index=True)

full_dataset_size.append(len(fart_uncurated))
dataset_stage.append('FlavorDB')
print(f"Added {len(flavor_db_subset)} molecules from FlavorDB")


# ============================================================================
# Process Tas2R Agonists
# ============================================================================

print("\n=== Processing Tas2R Agonists ===")
tas2r_agonists = pd.read_csv(DATASET_DIR / 'tas2r_agonists_db.csv')

# Extract relevant columns (preserve INCHIKEY if it exists)
cols_to_keep = ['Names', 'Canonical SMILES', 'CAS number']
if 'INCHIKEY' in tas2r_agonists.columns:
    cols_to_keep.append('INCHIKEY')
tas2r_agonists = tas2r_agonists[cols_to_keep].copy()

tas2r_agonists['Canonicalized Taste'] = 'bitter'
tas2r_agonists['Source'] = 'tas2r_agonists'
tas2r_agonists['Original Labels'] = 'Ligand to Tas2 Receptor'

tas2r_agonists = canonicalize_smiles_column(tas2r_agonists, 'Canonical SMILES')

# Extract identifiers from existing columns (INCHIKEY is already present)
tas2r_agonists = extract_identifiers_from_dataset(tas2r_agonists, inchikey_column='INCHIKEY')

tas2r_subset = tas2r_agonists[['Canonicalized SMILES', 'Canonicalized Taste', 'Original Labels', 'Source',
                                'PubChemID', 'InChI', 'INCHIKEY']]
tas2r_subset = tas2r_subset.reset_index(drop=True)
fart_uncurated = pd.concat([fart_uncurated, tas2r_subset], axis=0, ignore_index=True)

full_dataset_size.append(len(fart_uncurated))
dataset_stage.append('Tas2R Agonists')
print(f"Added {len(tas2r_subset)} molecules from Tas2R Agonists")


# ============================================================================
# Process PhytocompoundsDB (with CID)
# ============================================================================

print("\n=== Processing PhytocompoundsDB (with CID) ===")
phytocompounds_db = pd.read_csv(DATASET_DIR / 'phytocompounds_db_with_cid.csv')

def canonicalize_flavors_phytocompounds(flavor_string):
    assigned_flavors = []
    flavor_list = split_flavors(flavor_string)
    if ('bitter' in flavor_list) or ('bitter;' in flavor_list) or ('bitter, pungent' in flavor_list) or \
       ('bitter: pungent (scratchy)' in flavor_string) or ('bitter (electronic tongue)' in flavor_string) or \
       ('bitter ( electronic tongue)' in flavor_string) or ('bitter, sweet' in flavor_string) or \
       ('bitter, pungent (insufficient evidence)' in flavor_string) or ('bitter, astringent' in flavor_string):
        assigned_flavors.append('bitter')
    if ('sweet' in flavor_list) or ('sweet;' in flavor_list) or ('bitter, sweet' in flavor_string) or \
       ('sweet, bitter' in flavor_string):
        assigned_flavors.append('sweet')
    if ('sour' in flavor_list) or ('sour;' in flavor_list) or ('sour, astringent' in flavor_string):
        assigned_flavors.append('sour')
    if ('umami' in flavor_list) or ('umami-like' in flavor_list) or ('umami;' in flavor_list):
        assigned_flavors.append('umami')
    if len(assigned_flavors) == 0:
        assigned_flavors.append('undefined')
    return assigned_flavors

phytocompounds_db['Canonicalized Taste Intermediary'] = phytocompounds_db['Taste'].apply(
    lambda x: canonicalize_flavors_phytocompounds(x))
phytocompounds_db = tidy_canonical_flavors(phytocompounds_db)
phytocompounds_db = phytocompounds_db.dropna(subset=['Canonical SMILES'])
phytocompounds_db = canonicalize_smiles_column(phytocompounds_db, 'Canonical SMILES')
phytocompounds_db['Source'] = 'phytocompounds_db'
phytocompounds_db['Original Labels'] = phytocompounds_db['Taste']

# Extract identifiers from existing columns
phytocompounds_db = extract_identifiers_from_dataset(phytocompounds_db, cid_column='PubChem CID')

# Optionally enrich with API if enabled
if USE_PUBCHEM_API:
    phytocompounds_db = enrich_with_pubchem_api(phytocompounds_db, cid_column='PubChem CID')

phytocompounds_db_subset = phytocompounds_db[['Canonicalized SMILES', 'Canonicalized Taste', 'Original Labels', 'Source',
                                               'PubChemID', 'InChI', 'INCHIKEY']]
phytocompounds_db_subset = phytocompounds_db_subset.reset_index(drop=True)
fart_uncurated = pd.concat([fart_uncurated, phytocompounds_db_subset], axis=0, ignore_index=True)

full_dataset_size.append(len(fart_uncurated))
dataset_stage.append('PhytocompoundsDB')
print(f"Added {len(phytocompounds_db_subset)} molecules from PhytocompoundsDB")


# ============================================================================
# Process Umami DB
# ============================================================================

print("\n=== Processing Umami DB ===")
umami_db = pd.read_csv(DATASET_DIR / 'umami_db.csv')

umami_db['Canonicalized Taste'] = 'umami'
umami_db['Original Labels'] = 'Umami molecule from literature'
umami_db['Source'] = 'scifinder'

# Canonicalize SMILES (use 'Canonicalized SMILES' if it exists, otherwise assume it's already canonical)
if 'Canonicalized SMILES' in umami_db.columns:
    umami_db = canonicalize_smiles_column(umami_db, 'Canonicalized SMILES')
elif 'SMILES' in umami_db.columns:
    umami_db = canonicalize_smiles_column(umami_db, 'SMILES')
else:
    # If no SMILES column, skip canonicalization
    print("Warning: No SMILES column found in umami_db")

# Initialize PubChem columns
umami_db['PubChemID'] = None
umami_db['InChI'] = None
umami_db['INCHIKEY'] = None

umami_subset = umami_db[['Canonicalized SMILES', 'Canonicalized Taste', 'Original Labels', 'Source',
                         'PubChemID', 'InChI', 'INCHIKEY']]
umami_subset = umami_subset.reset_index(drop=True)
fart_uncurated = pd.concat([fart_uncurated, umami_subset], axis=0, ignore_index=True)

full_dataset_size.append(len(fart_uncurated))
dataset_stage.append('UmamiDB')
print(f"Added {len(umami_subset)} molecules from Umami DB")


# ============================================================================
# Process Sour DB
# ============================================================================

print("\n=== Processing Sour DB ===")
sour_db = pd.read_csv(DATASET_DIR / 'sour_db.csv')

# Filter for pKa1 between 2 and 7, temperature 15-30
sour_db['pka_value'] = pd.to_numeric(sour_db['pka_value'], errors='coerce')
sour_db['temperature'] = pd.to_numeric(sour_db['T'], errors='coerce')

sour_db_filtered = sour_db[sour_db['pka_type'] == 'pKa1']
sour_db_filtered = sour_db_filtered[
    (sour_db_filtered['temperature'] >= 15) & (sour_db_filtered['temperature'] <= 30)
]
sour_db_filtered = sour_db_filtered[
    (sour_db_filtered['pka_value'] >= 2) & (sour_db_filtered['pka_value'] < 7)
]

sour_db = canonicalize_smiles_column(sour_db_filtered, 'SMILES')
sour_db['Canonicalized Taste'] = 'sour'
sour_db['Source'] = 'IUPAC Dissocation Constants'
sour_db['Original Labels'] = 'pKa between 2 and 7'

# Extract identifiers from existing columns (InChI is already present)
sour_db = extract_identifiers_from_dataset(sour_db, inchi_column='InChI')

sour_db_subset = sour_db[['Canonicalized SMILES', 'Canonicalized Taste', 'Original Labels', 'Source',
                          'PubChemID', 'InChI', 'INCHIKEY']]
sour_db_subset = sour_db_subset.reset_index(drop=True)
fart_uncurated = pd.concat([fart_uncurated, sour_db_subset], axis=0, ignore_index=True)

full_dataset_size.append(len(fart_uncurated))
dataset_stage.append('sourDB')
print(f"Added {len(sour_db_subset)} molecules from Sour DB")


# ============================================================================
# Final Output
# ============================================================================

print("\n=== Final Dataset ===")
print(f"Total molecules: {len(fart_uncurated)}")
print(f"\nDataset size progression:")
for stage, size in zip(dataset_stage, full_dataset_size):
    print(f"  {stage}: {size}")

print(f"\nTaste distribution:")
print(fart_uncurated['Canonicalized Taste'].value_counts())

# Save to CSV
fart_uncurated.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved dataset to: {OUTPUT_FILE}")
print(f"Columns: {list(fart_uncurated.columns)}")
print(f"\nSummary of additional columns:")
print(f"  PubChemID: {fart_uncurated['PubChemID'].notna().sum()} / {len(fart_uncurated)} entries")
print(f"  InChI: {fart_uncurated['InChI'].notna().sum()} / {len(fart_uncurated)} entries")
print(f"  INCHIKEY: {fart_uncurated['INCHIKEY'].notna().sum()} / {len(fart_uncurated)} entries")

