import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", help="Path to food_context.jsonl")
    parser.add_argument("output_txt", help="Path to output .txt file for vanilla training")
    # Added argument to include the 10k PubChem sample
    parser.add_argument("--extra_smiles", help="Path to a raw .txt file (e.g. pubchem 10k) to append to the dataset", default=None)
    args = parser.parse_args()

    count_foodb = 0
    count_extra = 0
    
    print(f"Creating Baseline Dataset: {args.output_txt}")
    
    with open(args.output_txt, 'w') as outfile:
        # 1. Extract SMILES from FoodDB JSONL
        print(f"  Extracting from {args.input_jsonl}...")
        with open(args.input_jsonl, 'r') as infile:
            for line in infile:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if 'smiles' in data:
                            outfile.write(data['smiles'] + '\n')
                            count_foodb += 1
                    except json.JSONDecodeError:
                        continue
        
        # 2. Append SMILES from the 10k PubChem file (if provided)
        if args.extra_smiles and os.path.exists(args.extra_smiles):
            print(f"  Appending from {args.extra_smiles}...")
            with open(args.extra_smiles, 'r') as infile:
                for line in infile:
                    if line.strip():
                        outfile.write(line.strip() + '\n')
                        count_extra += 1
        elif args.extra_smiles:
            print(f"  Warning: Extra file {args.extra_smiles} not found!")

    total = count_foodb + count_extra
    print(f"\nDone! Combined Dataset Statistics:")
    print(f"  - FoodDB Molecules:   {count_foodb}")
    print(f"  - PubChem Molecules:  {count_extra}")
    print(f"  - Total for Baseline: {total}")

if __name__ == "__main__":
    main()