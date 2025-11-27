#!/usr/bin/env python3
"""
Extract random compounds from a text file (one SMILES per line) and optionally
augment them, writing the results to an output file.

Each line in the input file should contain one SMILES string (may be isomeric).
Each line in the output file contains one SMILES string.
"""

import argparse
import itertools
import math
import random
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PUBCHEM_DATA_DIR = PROJECT_ROOT / "data" / "pubchem"
FOODB_DATA_DIR = PROJECT_ROOT / "data" / "foodb"

try:
    from rdkit import Chem
except ImportError:
    print("Error: RDKit is not installed. Please install it with: pip install rdkit")
    sys.exit(1)


def control_smiles_duplication(random_smiles, duplicate_control=lambda x: 1):
    """
    Returns augmented SMILES with the number of duplicates controlled by the function duplicate_control.
    
    Taken from https://github.com/volkamerlab/maxsmi/blob/main/maxsmi/utils/utils_smiles.py
    
    Parameters
    ----------
    random_smiles : list
        A list of random SMILES, can be obtained by `smiles_to_random()`.
    duplicate_control : func, Optional, default: 1
        The number of times a SMILES will be duplicated, as function of the number of times
        it was included in `random_smiles`.
        This number is rounded up to the nearest integer.
    
    Returns
    -------
    list
        A list of random SMILES with duplicates.
    
    Notes
    -----
    When `duplicate_control=lambda x: 1`, then the returned list contains only unique SMILES.
    """
    counted_smiles = Counter(random_smiles)
    smiles_duplication = {
        smiles: math.ceil(duplicate_control(counted_smiles[smiles]))
        for smiles in counted_smiles
    }
    return list(
        itertools.chain.from_iterable(
            [[smiles] * smiles_duplication[smiles] for smiles in smiles_duplication]
        )
    )


def smiles_to_random(smiles, int_aug=50):
    """
    Takes a SMILES (not necessarily canonical) and returns `int_aug` random variations of this SMILES.
    
    Taken from https://github.com/volkamerlab/maxsmi/blob/main/maxsmi/utils/utils_smiles.py
    
    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.
    int_aug : int, Optional, default: 50
        The number of random SMILES generated.
    
    Returns
    -------
    list
        A list of `int_aug` random (may not be unique) SMILES or None if the initial SMILES is not valid.
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    else:
        if int_aug > 0:
            return [
                Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                for _ in range(int_aug)
            ]
        elif int_aug == 0:
            return [smiles]
        else:
            raise ValueError("int_aug must be greater or equal to zero.")


def augmentation_without_duplication(smiles, augmentation_number):
    """
    Takes a SMILES and returns a list of unique random SMILES.
    
    Taken from https://github.com/volkamerlab/maxsmi/blob/main/maxsmi/utils/utils_smiles.py
    
    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.
    augmentation_number : int
        The integer to generate the number of random SMILES.
    
    Returns
    -------
    list
        A list of unique random SMILES (no duplicates).
    """
    smiles_list = smiles_to_random(smiles, augmentation_number)
    if smiles_list is None:
        return None
    return control_smiles_duplication(smiles_list, lambda x: 1)


def convert_to_non_isomeric(smiles):
    """
    Convert a SMILES string to non-isomeric (remove stereochemistry).
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Non-isomeric SMILES string, or None if invalid
    """
    if not smiles or not smiles.strip():
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        # isomericSmiles=False to remove stereochemistry
        non_isomeric_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return non_isomeric_smiles
    except Exception:
        return None


def extract_random_smiles(input_file, output_file, num_compounds=1000, augment=False, num_augmentations=10, isomeric=True):
    """
    Extract random compounds from a text file (one SMILES per line) and optionally augment them.
    
    Args:
        input_file: Path to input text file (one SMILES per line)
        output_file: Path to output file
        num_compounds: Number of random compounds to extract
        augment: Whether to augment SMILES (generate multiple variations)
        num_augmentations: Number of augmented SMILES to generate per original (default: 10)
        isomeric: Whether to keep stereochemistry (True) or remove it (False). Default is True.
    """
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    print(f"Target: {num_compounds} random compounds")
    
    # Read all SMILES from file
    print("\nStep 1: Reading all SMILES from file...")
    compounds_with_smiles = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines_processed = 0
        for line in infile:
            lines_processed += 1
            smiles = line.strip()
            if smiles:  # Only collect non-empty lines
                compounds_with_smiles.append(smiles)
            
            if lines_processed % 10000 == 0:
                print(f"  Processed {lines_processed} lines, found {len(compounds_with_smiles)} with SMILES...", end='\r')
    
    print(f"\n  Total lines processed: {lines_processed}")
    print(f"  Compounds with SMILES: {len(compounds_with_smiles)}")
    
    if len(compounds_with_smiles) < num_compounds:
        print(f"\nWarning: Only {len(compounds_with_smiles)} compounds have SMILES, "
              f"but {num_compounds} were requested.")
        print(f"Will use all available compounds.")
        num_compounds = len(compounds_with_smiles)
    
    # Randomly sample compounds
    print(f"\nStep 2: Randomly sampling {num_compounds} compounds...")
    sampled = random.sample(compounds_with_smiles, num_compounds)
    
    # Convert to non-isomeric if requested
    if not isomeric:
        print(f"\nStep 3: Converting SMILES to non-isomeric (removing stereochemistry)...")
        processed_smiles = []
        failed_count = 0
        
        for i, smiles in enumerate(sampled):
            non_isomeric = convert_to_non_isomeric(smiles)
            if non_isomeric:
                processed_smiles.append(non_isomeric)
            else:
                failed_count += 1
                print(f"  Warning: Failed to convert SMILES: {smiles[:50]}...")
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{num_compounds} compounds...", end='\r')
        
        print(f"\n  Successfully converted: {len(processed_smiles)}")
        print(f"  Failed to convert: {failed_count}")
        sampled = processed_smiles
    
    # Augment SMILES if requested
    if augment:
        step_num = 4 if not isomeric else 3
        print(f"\nStep {step_num}: Generating {num_augmentations} augmented SMILES per original...")
        all_smiles = []
        augmentation_failed = 0
        
        for i, original_smiles in enumerate(sampled):
            # Keep generating until we have exactly num_augmentations unique valid SMILES
            unique_augmented = set()
            attempts = 0
            max_attempts = num_augmentations * 20  # Try up to 20x to get unique ones
            
            while len(unique_augmented) < num_augmentations and attempts < max_attempts:
                # Generate more random SMILES than we need to increase chances of getting unique ones
                augmented_list = augmentation_without_duplication(original_smiles, num_augmentations * 3)
                
                if augmented_list is None:
                    # If augmentation fails, use the original
                    unique_augmented.add(original_smiles)
                    augmentation_failed += 1
                    break
                
                # Validate and add unique valid SMILES
                for aug_smiles in augmented_list:
                    # Validate that it's a valid SMILES, but keep it as-is
                    mol = Chem.MolFromSmiles(aug_smiles)
                    if mol is not None:
                        unique_augmented.add(aug_smiles)
                        if len(unique_augmented) >= num_augmentations:
                            break
                
                attempts += 1
            
            # Convert to list for final processing
            augmented_list_final = list(unique_augmented)
            original_count = len(augmented_list_final)
            
            # If we still don't have enough unique ones, pad by repeating existing ones
            # (This handles cases where molecules have very few unique SMILES representations)
            if original_count == 0:
                # If we have no valid augmented SMILES, use the original
                augmented_list_final = [original_smiles] * num_augmentations
            elif original_count < num_augmentations:
                # Cycle through the unique SMILES we have to reach the target count
                while len(augmented_list_final) < num_augmentations:
                    idx = len(augmented_list_final) % original_count
                    augmented_list_final.append(augmented_list_final[idx])
            
            # Add exactly num_augmentations SMILES
            all_smiles.extend(augmented_list_final[:num_augmentations])
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(sampled)} compounds, generated {len(all_smiles)} total SMILES...", end='\r')
        
        print(f"\n  Total augmented SMILES generated: {len(all_smiles)}")
        print(f"  Compounds with failed augmentation: {augmentation_failed}")
        
        # Write all augmented SMILES to file
        step_num = 5 if not isomeric else 4
        print(f"\nStep {step_num}: Writing augmented SMILES to file...")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for smiles in all_smiles:
                outfile.write(smiles + '\n')
        
        total_count = len(all_smiles)
    else:
        # Write sampled SMILES to file
        step_num = 4 if not isomeric else 3
        print(f"\nStep {step_num}: Writing sampled SMILES to file...")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for smiles in sampled:
                outfile.write(smiles + '\n')
        
        total_count = len(sampled)
    
    print(f"\n\nCompleted!")
    print(f"  Total SMILES written: {total_count}")
    print(f"  Output file: {output_file}")
    
    return total_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract random compounds from a text file (one SMILES per line) and optionally augment them"
    )
    parser.add_argument(
        "-n", "--num-compounds",
        type=int,
        help="Number of random compounds to extract (default: prompt for input)"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Input text file (one SMILES per line). If not provided, will prompt for dataset choice."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output text file (default: data/<dataset>/<dataset>_<N>_smiles.txt)"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Generate augmented SMILES (10 per original)"
    )
    
    args = parser.parse_args()
    
    # Determine input file
    if args.input:
        # Use provided input file
        input_file = Path(args.input)
        dataset_name = "custom"
    else:
        # Ask user to choose dataset
        try:
            print("\nChoose dataset:")
            print("  (1) pubchem")
            print("  (2) foodb")
            choice = input("Enter choice (1 or 2): ").strip()
            
            if choice == "1":
                dataset_name = "pubchem"
                input_file = PUBCHEM_DATA_DIR / "pubchem_smiles_all.txt"
            elif choice == "2":
                dataset_name = "foodb"
                input_file = FOODB_DATA_DIR / "foodb_smiles_all.txt"
            else:
                print("Error: Invalid choice. Please enter 1 or 2.")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\nCancelled by user")
            sys.exit(1)
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)
    
    # Get number of compounds
    if args.num_compounds is not None:
        num_compounds = args.num_compounds
    else:
        try:
            num_compounds = int(input("Enter the number of SMILES to sample: "))
            if num_compounds <= 0:
                print("Error: Number must be positive")
                sys.exit(1)
        except ValueError:
            print("Error: Invalid number")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nCancelled by user")
            sys.exit(1)
    
    # Ask about augmentation
    if args.augment:
        # Command-line flag was provided
        augment = True
    else:
        # Ask the user (unless --augment was explicitly set)
        try:
            augment_response = input("Do you want to augment SMILES? (Y/n): ").strip().upper()
            augment = augment_response in ['Y', 'YES', '']
        except KeyboardInterrupt:
            print("\nCancelled by user")
            sys.exit(1)
    
    # Ask about isomeric (default is True/isomeric)
    try:
        isomeric_response = input("Keep stereochemistry (isomeric)? (Y/n, default=Y): ").strip().upper()
        isomeric = isomeric_response not in ['N', 'NO']
    except KeyboardInterrupt:
        print("\nCancelled by user")
        sys.exit(1)
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        # Choose output directory based on dataset
        if dataset_name == "pubchem":
            output_dir = PUBCHEM_DATA_DIR
        elif dataset_name == "foodb":
            output_dir = FOODB_DATA_DIR
        else:  # custom
            output_dir = PROCESSED_DATA_DIR
        
        # Build filename with optional nonisomeric and augmented suffixes
        nonisomeric_suffix = "_nonisomeric" if not isomeric else ""
        
        if augment:
            output_file = output_dir / f"{dataset_name}_{num_compounds}{nonisomeric_suffix}_augmented_smiles.txt"
        else:
            output_file = output_dir / f"{dataset_name}_{num_compounds}{nonisomeric_suffix}_smiles.txt"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        count = extract_random_smiles(
            input_file, 
            output_file, 
            num_compounds=num_compounds,
            augment=augment,
            num_augmentations=10,
            isomeric=isomeric
        )
        if augment:
            print(f"\nSuccessfully created {output_file} with {count} augmented SMILES ({num_compounds} originals × 10)")
        else:
            print(f"\nSuccessfully created {output_file} with {count} sampled SMILES")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

