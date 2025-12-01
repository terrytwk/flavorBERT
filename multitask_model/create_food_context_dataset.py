#!/usr/bin/env python3
"""
Create a dataset that maps SMILES to food contexts (co-occurrence information).

Output format (JSON Lines):
{"smiles": "CC(=O)O", "foods": ["apple", "wine", "vinegar"], "food_indices": [0, 45, 123]}

This can be used for multi-task learning: MLM + food co-occurrence prediction.
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
from rdkit import Chem


def canonicalize_smiles(smiles):
    """Convert SMILES to canonical form. Returns None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def load_compound_smiles_mapping(foodb_dir):
    """Load compound_id -> canonical SMILES mapping from Compound.json."""
    compound_json = os.path.join(foodb_dir, 'Compound.json')
    compound_id_to_smiles = {}
    
    print("Loading compound SMILES mapping...")
    with open(compound_json, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                compound_id = record.get('id')
                if not compound_id or 'moldb_smiles' not in record:
                    continue
                
                smiles = str(record['moldb_smiles']).strip()
                if not smiles or smiles == 'None' or smiles == '':
                    continue
                
                canonical = canonicalize_smiles(smiles)
                if canonical is not None:
                    compound_id_to_smiles[compound_id] = canonical
            except Exception as e:
                continue
    
    print(f"  Loaded {len(compound_id_to_smiles)} compounds with valid SMILES")
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
            
            try:
                record = json.loads(line)
                food_id = record.get('id')
                food_name = record.get('name')
                if food_id and food_name:
                    food_id_to_name[food_id] = food_name
            except:
                continue
    
    print(f"  Loaded {len(food_id_to_name)} foods")
    return food_id_to_name


def build_smiles_to_foods_mapping(foodb_dir, compound_id_to_smiles, food_id_to_name):
    """
    Build mapping from SMILES to list of foods containing that compound.
    
    Returns:
        smiles_to_foods: dict mapping SMILES -> set of food names
    """
    content_json = os.path.join(foodb_dir, 'Content.json')
    smiles_to_foods = defaultdict(set)
    
    print("Loading compound-food relationships...")
    with open(content_json, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                if record.get('source_type') != 'Compound':
                    continue
                
                compound_id = record.get('source_id')
                food_id = record.get('food_id')
                
                if not (compound_id and food_id):
                    continue
                
                smiles = compound_id_to_smiles.get(compound_id)
                food_name = food_id_to_name.get(food_id)
                
                if smiles and food_name:
                    smiles_to_foods[smiles].add(food_name)
            except:
                continue
    
    print(f"  Found {len(smiles_to_foods)} compounds with food associations")
    return smiles_to_foods


def create_food_vocabulary(smiles_to_foods, min_occurrences=2):
    """
    Create a vocabulary of foods, filtering out rare foods.
    
    Returns:
        food_to_idx: dict mapping food name -> index
        idx_to_food: dict mapping index -> food name
    """
    # Count food occurrences
    food_counts = defaultdict(int)
    for foods in smiles_to_foods.values():
        for food in foods:
            food_counts[food] += 1
    
    # Filter and sort foods
    frequent_foods = sorted([
        food for food, count in food_counts.items() 
        if count >= min_occurrences
    ])
    
    # Create vocabulary
    food_to_idx = {food: idx for idx, food in enumerate(frequent_foods)}
    idx_to_food = {idx: food for food, idx in food_to_idx.items()}
    
    print(f"  Food vocabulary size: {len(food_to_idx)} (min occurrences: {min_occurrences})")
    return food_to_idx, idx_to_food


def create_food_context_dataset(foodb_dir, output_file, min_food_occurrences=2):
    """
    Create the complete food context dataset.
    
    Args:
        foodb_dir: Path to FoodDB JSON files
        output_file: Output JSONL file path
        min_food_occurrences: Minimum times a food must appear to be included
    """
    # Load mappings
    compound_id_to_smiles = load_compound_smiles_mapping(foodb_dir)
    food_id_to_name = load_food_name_mapping(foodb_dir)
    smiles_to_foods = build_smiles_to_foods_mapping(
        foodb_dir, compound_id_to_smiles, food_id_to_name
    )
    
    # Create food vocabulary
    food_to_idx, idx_to_food = create_food_vocabulary(smiles_to_foods, min_food_occurrences)
    
    # Write dataset
    print(f"\nWriting dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for smiles, foods in smiles_to_foods.items():
            # Filter foods to only include those in vocabulary
            valid_foods = [food for food in foods if food in food_to_idx]
            
            if valid_foods:  # Only include compounds with at least one valid food
                food_indices = sorted([food_to_idx[food] for food in valid_foods])
                
                record = {
                    "smiles": smiles,
                    "foods": sorted(valid_foods),
                    "food_indices": food_indices
                }
                f.write(json.dumps(record) + '\n')
    
    # Write vocabulary
    vocab_file = output_file.replace('.jsonl', '_food_vocab.json')
    print(f"Writing food vocabulary to {vocab_file}...")
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump({
            "food_to_idx": food_to_idx,
            "idx_to_food": idx_to_food,
            "vocab_size": len(food_to_idx)
        }, f, indent=2)
    
    print(f"\n✓ Created dataset with {len(smiles_to_foods)} compound-food associations")
    print(f"✓ Food vocabulary size: {len(food_to_idx)}")
    return food_to_idx


def main():
    parser = argparse.ArgumentParser(
        description="Create food context dataset from FoodDB for multi-task learning"
    )
    parser.add_argument(
        "--foodb_dir",
        type=str,
        default="chemberta/data/foodb",
        help="Path to FoodDB JSON files directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chemberta/data/foodb_context.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--min_food_occurrences",
        type=int,
        default=2,
        help="Minimum times a food must appear to be included in vocabulary"
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    create_food_context_dataset(
        args.foodb_dir,
        args.output,
        args.min_food_occurrences
    )
    
    print("\n✓ Done! You can now use this dataset for multi-task training.")


if __name__ == "__main__":
    main()

