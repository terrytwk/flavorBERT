import argparse
import multiprocessing
import random
import os
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm

def process_smiles(smiles):
    """
    Worker function to canonicalize a single SMILES string.
    Returns the canonical string or None if invalid.
    """
    smiles = smiles.strip()
    if not smiles:
        return None
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def main():
    parser = argparse.ArgumentParser(description="Sample and Canonicalize SMILES")
    parser.add_argument("input_file", help="Path to the large input SMILES file")
    parser.add_argument("num_samples", type=int, help="Number of SMILES to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU cores (default: all)")
    
    args = parser.parse_args()
    random.seed(args.seed)

    input_path = Path(args.input_file)
    
    # 1. Determine Output Path (Same directory as input)
    # Name format: original_name_sampled_100000.txt
    output_filename = f"{input_path.stem}_sampled_{args.num_samples}{input_path.suffix}"
    output_path = input_path.parent / output_filename
    
    print(f"Input File:  {input_path}")
    print(f"Target Size: {args.num_samples:,}")
    print(f"Output File: {output_path}")

    # 2. Read lines
    # For 10M lines, this takes ~500MB RAM, which is negligible for H200 nodes.
    print(f"\n[1/3] Reading file into memory...")
    with open(input_path, 'r', encoding='utf-8') as f:
        # Filter empty lines immediately
        lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(lines)
    print(f"      File contains {total_lines:,} lines.")

    # 3. Sample
    if args.num_samples >= total_lines:
        print(f"      Requested samples ({args.num_samples:,}) >= Total lines. Using all lines.")
        sampled_raw = lines
    else:
        print(f"      Randomly sampling {args.num_samples:,} lines...")
        sampled_raw = random.sample(lines, args.num_samples)

    # 4. Canonicalize in Parallel
    print(f"\n[2/3] Canonicalizing with {multiprocessing.cpu_count() if args.workers is None else args.workers} cores...")
    
    valid_smiles = []
    
    # Use multiprocessing pool
    pool = multiprocessing.Pool(processes=args.workers)
    
    # We use tqdm to show progress
    results = list(tqdm(pool.imap(process_smiles, sampled_raw, chunksize=1000), total=len(sampled_raw)))
    
    pool.close()
    pool.join()

    # Filter out Nones (failed canonicalizations)
    valid_smiles = [s for s in results if s is not None]
    
    # 5. Write Output
    print(f"\n[3/3] Writing to disk...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for smiles in valid_smiles:
            f.write(smiles + '\n')

    print("\n" + "="*40)
    print("COMPLETE")
    print(f"Requested: {args.num_samples:,}")
    print(f"Valid:     {len(valid_smiles):,} ({(len(valid_smiles)/len(sampled_raw))*100:.1f}%)")
    print(f"Dropped:   {len(sampled_raw) - len(valid_smiles):,}")
    print(f"Saved to:  {output_path}")
    print("="*40)

if __name__ == "__main__":
    main()