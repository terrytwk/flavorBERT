"""
Extract SMILES strings from FoodDB compounds and create concentration matrices.

FoodDB data format:
- Compound.json: JSONL file with compound data, including 'moldb_smiles' field
- Food.json: JSONL file with food data, including 'name' and 'id' fields
- Content.json: JSONL file linking compounds to foods via:
  - source_id: compound_id (when source_type is "Compound")
  - food_id: food identifier
  - standard_content: concentration value (in mg/100g)
"""

import pandas as pd
import numpy as np
import json
import os
from rdkit import Chem


def canonicalize_smiles(smiles):
    """Convert SMILES to canonical form. Returns None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_compound_smiles_mapping(foodb_dir, canonicalize=True):
    """Load compound_id -> SMILES mapping from Compound.json."""
    compound_json = os.path.join(foodb_dir, 'Compound.json')
    compound_id_to_smiles = {}
    
    print("Loading compound SMILES mapping...")
    with open(compound_json, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)
            compound_id = record.get('id')
            if not compound_id or 'moldb_smiles' not in record or not record['moldb_smiles']:
                continue
            
            smiles = str(record['moldb_smiles']).strip()
            if not smiles or smiles == 'None' or smiles == '':
                continue
            
            # Check for structural characters
            if not any(c in smiles for c in ['(', ')', '[', ']', '@']):
                continue
            
            if canonicalize:
                canonical = canonicalize_smiles(smiles)
                if canonical is not None:
                    compound_id_to_smiles[compound_id] = canonical
            else:
                compound_id_to_smiles[compound_id] = smiles
    
    print(f"  Loaded {len(compound_id_to_smiles)} compounds")
    return compound_id_to_smiles


def load_food_name_mapping(foodb_dir):
    """Load food_id -> food_name mapping from Food.json."""
    food_json = os.path.join(foodb_dir, 'Food.json')
    food_id_to_name = {}
    
    print("Loading food name mapping...")
    with open(food_json, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)
            food_id = record.get('id')
            food_name = record.get('name')
            if food_id and food_name:
                food_id_to_name[food_id] = food_name
    
    print(f"  Loaded {len(food_id_to_name)} foods")
    return food_id_to_name


def create_concentration_matrix(foodb_dir=None, canonicalize=True, 
                                compound_id_to_smiles=None, food_id_to_name=None):
    """
    Create concentration matrix: rows=SMILES, columns=foods.
    Values are log-transformed (log1p) to handle scale differences.
    """
    if foodb_dir is None:
        foodb_dir = '../data/foodb'
    
    # Step 1: Load compound and food mappings if not provided
    if compound_id_to_smiles is None:
        compound_id_to_smiles = load_compound_smiles_mapping(foodb_dir, canonicalize)
    if food_id_to_name is None:
        food_id_to_name = load_food_name_mapping(foodb_dir)
    
    # Step 2: Build concentration dictionary from Content.json
    content_json = os.path.join(foodb_dir, 'Content.json')
    concentration_dict = {}  # (smiles, food_name) -> concentration
    
    print("Loading concentration data...")
    with open(content_json, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)
            if record.get('source_type') != 'Compound':
                continue
            
            compound_id = record.get('source_id')
            food_id = record.get('food_id')
            content = record.get('standard_content')
            
            if not (compound_id and food_id and content):
                continue
            
            smiles = compound_id_to_smiles.get(compound_id)
            if not smiles:
                continue
            
            try:
                content_val = float(content)
                if content_val <= 0:
                    continue
            except (ValueError, TypeError):
                continue
            
            food_name = food_id_to_name.get(food_id)
            if not food_name:
                continue
            
            key = (smiles, food_name)
            concentration_dict[key] = concentration_dict.get(key, 0) + content_val
    
    print(f"  Found {len(concentration_dict)} compound-food pairs")
    
    # Step 3: Build and normalize matrix
    if not concentration_dict:
        print("  WARNING: No concentration data found!")
        return pd.DataFrame()
    
    all_smiles = sorted(set(smiles for smiles, _ in concentration_dict.keys()))
    all_foods = sorted(set(food for _, food in concentration_dict.keys()))
    
    matrix_data = {food: [0.0] * len(all_smiles) for food in all_foods}
    for (smiles, food), concentration in concentration_dict.items():
        smiles_idx = all_smiles.index(smiles)
        matrix_data[food][smiles_idx] = concentration
    
    df = pd.DataFrame(matrix_data, index=all_smiles)
    
    # Apply log-transform to handle scale differences while preserving relationships
    df = np.log1p(df)  # log(1+x) to handle zeros
    
    df = df.reset_index().rename(columns={'index': 'smiles'})
    
    print(f"  Matrix: {df.shape[0]} compounds x {df.shape[1]-1} foods")
    print(f"  Normalization: log-transform (log1p)")
    
    return df


if __name__ == "__main__":
    foodb_dir = '../data/foodb'
    
    compound_id_to_smiles = load_compound_smiles_mapping(foodb_dir, canonicalize=True)
    food_id_to_name = load_food_name_mapping(foodb_dir)
    df = create_concentration_matrix(foodb_dir, canonicalize=True, 
                                     compound_id_to_smiles=compound_id_to_smiles,
                                     food_id_to_name=food_id_to_name)
    
    print(df.head())
    df.to_csv('../data/foodDataStructures/concentrationMatrix.csv', index=False)
    print("Saved concentration matrix to concentrationMatrix.csv")