"""
Script to fetch SMILES string from PubChem using a PubChem ID (CID).

Usage:
    # Interactive mode - prompts for PubChem ID
    python get_smiles_from_pubchem.py
    
    # Command-line mode - pass PubChem ID as argument
    python get_smiles_from_pubchem.py 12345
    
    # Batch mode - provide multiple IDs
    python get_smiles_from_pubchem.py 12345 67890 11111
"""

import sys
import argparse
from typing import Optional, List, Dict
try:
    import pubchempy as pcp
except ImportError:
    print("ERROR: pubchempy not installed. Install it with: pip install pubchempy")
    sys.exit(1)


def get_smiles_from_pubchem(cid: str, include_isomeric: bool = False) -> Dict[str, Optional[str]]:
    """
    Fetch SMILES string from PubChem using a PubChem ID (CID).
    
    Args:
        cid: PubChem Compound ID (CID) as string or int
        include_isomeric: If True, also return isomeric SMILES
    
    Returns:
        Dictionary with:
            - 'cid': The PubChem ID
            - 'canonical_smiles': Canonical SMILES (if available)
            - 'isomeric_smiles': Isomeric SMILES (if include_isomeric=True and available)
            - 'iupac_name': IUPAC name (if available)
            - 'error': Error message (if any)
    """
    result = {
        'cid': str(cid),
        'canonical_smiles': None,
        'isomeric_smiles': None,
        'iupac_name': None,
        'error': None
    }
    
    try:
        # Convert CID to int if it's a string
        cid_int = int(cid)
        
        # Use get_properties for more reliable SMILES retrieval
        properties = ['SMILES', 'IUPACName']
        if include_isomeric:
            properties.append('IsomericSMILES')
        
        prop_data = pcp.get_properties(properties, cid_int, 'cid')
        
        if not prop_data or len(prop_data) == 0:
            result['error'] = f"PubChem ID {cid} not found in PubChem database"
            return result
        
        prop_dict = prop_data[0]
        
        # Get canonical SMILES from SMILES property
        if 'SMILES' in prop_dict and prop_dict['SMILES']:
            result['canonical_smiles'] = prop_dict['SMILES']
        
        # Get isomeric SMILES if requested
        if include_isomeric:
            if 'IsomericSMILES' in prop_dict and prop_dict['IsomericSMILES']:
                result['isomeric_smiles'] = prop_dict['IsomericSMILES']
            elif result['canonical_smiles']:
                # Fallback to canonical if isomeric not available
                result['isomeric_smiles'] = result['canonical_smiles']
        
        # Get IUPAC name if available
        if 'IUPACName' in prop_dict and prop_dict['IUPACName']:
            result['iupac_name'] = prop_dict['IUPACName']
        
        # If we still don't have SMILES, that's an error
        if not result['canonical_smiles']:
            result['error'] = f"Could not retrieve SMILES for PubChem ID {cid}"
            
    except pcp.BadRequestError as e:
        result['error'] = f"Bad request error: Invalid PubChem ID format"
    except (pcp.NotFoundError, IndexError, KeyError):
        result['error'] = f"PubChem ID {cid} not found in PubChem database"
    except ValueError as e:
        result['error'] = f"Invalid PubChem ID format: {cid} (must be numeric)"
    except Exception as e:
        result['error'] = f"Error fetching data: {str(e)}"
    
    return result


def print_result(result: Dict[str, Optional[str]], verbose: bool = False, simple: bool = False):
    """Print the result in a formatted way."""
    cid = result['cid']
    
    if result['error']:
        if simple:
            # In simple mode, print nothing on error (or print empty line)
            pass
        else:
            print(f"\n❌ Error for PubChem ID {cid}:")
            print(f"   {result['error']}")
        return
    
    if simple:
        # Simple mode: just print the SMILES
        if result['canonical_smiles']:
            print(result['canonical_smiles'])
        return
    
    print(f"\n✅ PubChem ID: {cid}")
    
    if result['canonical_smiles']:
        print(f"   Canonical SMILES: {result['canonical_smiles']}")
    
    if result['isomeric_smiles']:
        print(f"   Isomeric SMILES:  {result['isomeric_smiles']}")
    
    if verbose:
        if result['iupac_name']:
            print(f"   IUPAC Name:       {result['iupac_name']}")
        
        # Check if SMILES is isomeric
        if result['canonical_smiles']:
            is_isomeric = '@' in result['canonical_smiles'] or '/' in result['canonical_smiles'] or '\\' in result['canonical_smiles']
            if is_isomeric:
                print(f"   Contains stereochemistry: Yes")
            else:
                print(f"   Contains stereochemistry: No")


def interactive_mode(include_isomeric: bool = False, verbose: bool = False, simple: bool = False):
    """Interactive mode - prompts user for PubChem IDs."""
    if not simple:
        print("=" * 80)
        print("PubChem ID to SMILES Converter")
        print("=" * 80)
        print("Enter PubChem IDs (one per line). Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            user_input = input("PubChem ID: " if not simple else "").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                if not simple:
                    print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            result = get_smiles_from_pubchem(user_input, include_isomeric=include_isomeric)
            print_result(result, verbose=verbose, simple=simple)
            if not simple:
                print()  # Empty line for readability
            
        except KeyboardInterrupt:
            if not simple:
                print("\n\nInterrupted by user. Goodbye!")
            break
        except EOFError:
            if not simple:
                print("\n\nGoodbye!")
            break


def batch_mode(cids: List[str], include_isomeric: bool = False, verbose: bool = False, simple: bool = False):
    """Batch mode - process multiple PubChem IDs from command line."""
    if not simple:
        print("=" * 80)
        print(f"Fetching SMILES for {len(cids)} PubChem ID(s)")
        print("=" * 80)
    
    for i, cid in enumerate(cids, 1):
        if not simple and len(cids) > 1:
            print(f"\n[{i}/{len(cids)}] Processing CID {cid}...")
        
        result = get_smiles_from_pubchem(cid, include_isomeric=include_isomeric)
        print_result(result, verbose=verbose, simple=simple)
    
    if not simple and len(cids) > 1:
        print("\n" + "=" * 80)
        print("Summary:")
        successful = sum(1 for cid in cids if get_smiles_from_pubchem(cid)['canonical_smiles'])
        print(f"  Successfully fetched: {successful}/{len(cids)}")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch SMILES string from PubChem using PubChem ID (CID)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode - prompts for PubChem ID
  python get_smiles_from_pubchem.py
  
  # Fetch SMILES for a single CID
  python get_smiles_from_pubchem.py 12345
  
  # Fetch SMILES for multiple CIDs
  python get_smiles_from_pubchem.py 12345 67890 11111
  
  # Include isomeric SMILES and IUPAC name
  python get_smiles_from_pubchem.py --isomeric --verbose 12345
        """
    )
    
    parser.add_argument(
        'cids',
        nargs='*',
        help='PubChem Compound ID(s) (CID). If not provided, runs in interactive mode.'
    )
    
    parser.add_argument(
        '--isomeric',
        action='store_true',
        help='Also fetch and display isomeric SMILES (includes stereochemistry)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show additional information (IUPAC name, stereochemistry info)'
    )
    
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Output only SMILES strings (one per line, useful for scripting)'
    )
    
    args = parser.parse_args()
    
    # If CIDs provided, use batch mode; otherwise interactive mode
    if args.cids:
        batch_mode(args.cids, include_isomeric=args.isomeric, verbose=args.verbose, simple=args.simple)
    else:
        interactive_mode(include_isomeric=args.isomeric, verbose=args.verbose, simple=args.simple)


if __name__ == "__main__":
    main()

