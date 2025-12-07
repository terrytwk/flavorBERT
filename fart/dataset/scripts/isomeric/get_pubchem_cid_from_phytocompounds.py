"""
Script to extract PubChem CID from phytocompounds database.

This script reads phytocompounds_db.csv and for each compound:
1. Uses the phytocompounds ID to construct the compound page URL
2. Scrapes the page to find the PubChem CID
3. Creates a new CSV file with the PubChem CID column added

Usage:
    python get_pubchem_cid_from_phytocompounds.py
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import re
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


def get_pubchem_cid_from_page(compound_id, session=None):
    """
    Fetch PubChem CID from a phytocompounds compound page.
    
    Args:
        compound_id: The phytocompounds ID (e.g., 'PMTDB00001')
        session: Optional requests.Session object for connection pooling
    
    Returns:
        PubChem CID as string, or None if not found
    """
    compound_url = f"https://plantmoleculartastedb.org/compound.php?id={compound_id}"
    
    # Use session if provided for connection pooling, otherwise use requests directly
    request_func = session.get if session else requests.get
    
    try:
        response = request_func(compound_url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Method 1: Look for links to pubchem.ncbi.nlm.nih.gov/compound/ (most reliable)
        # Based on actual page structure: <a href="https://pubchem.ncbi.nlm.nih.gov/compound/261491">261491</a>
        pubchem_links = soup.find_all("a", href=re.compile(r"pubchem\.ncbi\.nlm\.nih\.gov/compound/", re.I))
        for link in pubchem_links:
            href = link.get("href", "")
            # Skip if URL contains "absent" (no PubChem CID available)
            if "absent" in href.lower():
                continue
            # Extract CID from URL: https://pubchem.ncbi.nlm.nih.gov/compound/12345
            match = re.search(r"/compound/(\d+)", href)
            if match:
                cid = match.group(1)
                # Verify it's a valid CID (should be digits)
                if cid.isdigit():
                    return cid
            # Also check link text in case URL format is different
            link_text = link.get_text().strip()
            # Skip if link text is "absent" or not a valid number
            if link_text.lower() == "absent":
                continue
            if link_text.isdigit() and len(link_text) >= 4:
                return link_text
        
        # Method 2: Look for text "PubChem CID:" followed by a link or number
        # Based on actual page structure: "PubChem CID: <a href="...">261491</a>"
        # Find all text nodes containing "PubChem CID"
        for text_node in soup.find_all(string=re.compile(r"PubChem\s+CID", re.I)):
            # Get the parent element
            parent = text_node.parent
            if parent:
                # Look for the next link or number after "PubChem CID:"
                # Check if there's a link in the same parent
                link = parent.find("a", href=re.compile(r"pubchem", re.I))
                if link:
                    href = link.get("href", "")
                    match = re.search(r"/compound/(\d+)", href)
                    if match:
                        return match.group(1)
                    # Or use link text
                    link_text = link.get_text().strip()
                    if link_text.isdigit():
                        return link_text
                
                # Look for number in the parent's text after "PubChem CID:"
                parent_text = parent.get_text()
                match = re.search(r"PubChem\s+CID[:\s]+(\d+)", parent_text, re.I)
                if match:
                    return match.group(1)
        
        # Method 3: Search in the content div (where identifiers are typically shown)
        # Based on structure: <div id="content"> contains the PubChem CID link
        content_div = soup.find("div", id="content")
        if content_div:
            # Look for PubChem links in the content div
            pubchem_links = content_div.find_all("a", href=re.compile(r"pubchem", re.I))
            for link in pubchem_links:
                href = link.get("href", "")
                match = re.search(r"/compound/(\d+)", href)
                if match:
                    return match.group(1)
                link_text = link.get_text().strip()
                if link_text.isdigit() and len(link_text) >= 4:
                    return link_text
        
        # Method 4: Fallback - search entire page text for "PubChem CID: 12345" pattern
        text_content = soup.get_text()
        match = re.search(r"PubChem\s+CID[:\s]+(\d+)", text_content, re.I)
        if match:
            return match.group(1)
        
        # Method 5: Look for any link to pubchem (broader search)
        all_pubchem_links = soup.find_all("a", href=re.compile(r"pubchem", re.I))
        for link in all_pubchem_links:
            href = link.get("href", "")
            # Try to extract CID from various URL formats
            match = re.search(r"(?:compound/|cid=)(\d+)", href, re.I)
            if match:
                return match.group(1)
        
        return None
        
    except requests.Timeout:
        print(f"Timeout fetching page for {compound_id}")
        return None
    except requests.HTTPError as e:
        if e.response.status_code == 429:
            print(f"Rate limited for {compound_id}. Consider increasing --delay")
        else:
            print(f"HTTP error {e.response.status_code} for {compound_id}: {e}")
        return None
    except requests.RequestException as e:
        print(f"Error fetching page for {compound_id}: {e}")
        return None
    except Exception as e:
        print(f"Error parsing page for {compound_id}: {e}")
        return None


def process_phytocompounds_csv(input_csv_path, output_csv_path=None, batch_size=100, delay=0.1, max_workers=10, verbose=False):
    """
    Process phytocompounds CSV file and add PubChem CID column using concurrent requests.
    
    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path to output CSV file (default: adds '_with_cid' to input name)
        batch_size: Number of rows to process before saving (for checkpointing)
        delay: Delay between batches in seconds (to be respectful to the server)
        max_workers: Number of concurrent threads for parallel requests
        verbose: If True, print detailed progress for each compound
    """
    # Read the CSV file
    print(f"Reading {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    # Check if 'id' column exists
    if 'id' not in df.columns:
        print("Error: 'id' column not found in CSV file.")
        print(f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)
    
    # Create output path if not provided
    if output_csv_path is None:
        base_name = os.path.splitext(input_csv_path)[0]
        output_csv_path = f"{base_name}_with_cid.csv"
    
    # Check if output file exists (for resuming)
    processed_ids = set()
    if os.path.exists(output_csv_path):
        print(f"Found existing output file: {output_csv_path}")
        response = input("Do you want to resume from existing file? (y/n): ").strip().lower()
        if response == 'y':
            df_existing = pd.read_csv(output_csv_path)
            processed_ids = set(df_existing[df_existing['PubChem CID'].notna()]['id'].dropna())
            df_result = df_existing.copy()
            print(f"Resuming: {len(processed_ids)} compounds already processed")
        else:
            df_result = df.copy()
            df_result['PubChem CID'] = None
    else:
        df_result = df.copy()
        df_result['PubChem CID'] = None
    
    # Get unique compound IDs (in case there are duplicates)
    unique_ids = [uid for uid in df['id'].unique() if pd.notna(uid) and uid not in processed_ids]
    print(f"Processing {len(unique_ids)} unique compounds with {max_workers} concurrent workers...")
    
    # Create a mapping of ID to PubChem CID (thread-safe with lock)
    # Initialize with already processed IDs from existing file
    id_to_cid = {}
    if len(processed_ids) > 0:
        mask = df_result['id'].isin(processed_ids) & df_result['PubChem CID'].notna()
        existing_cids = df_result[mask]
        id_to_cid = {row['id']: row['PubChem CID'] for _, row in existing_cids.iterrows()}
    
    id_to_cid_lock = Lock()
    completed_count = [0]  # Use list for mutable reference in nested function
    
    # Create a session for connection pooling
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; research script)'
    })
    
    def fetch_cid(compound_id):
        """Wrapper function for concurrent execution"""
        try:
            cid = get_pubchem_cid_from_page(compound_id, session=session)
            with id_to_cid_lock:
                id_to_cid[compound_id] = cid
                completed_count[0] += 1
            if verbose and cid:
                print(f"  {compound_id} -> CID: {cid}")
            return compound_id, cid
        except Exception as e:
            if verbose:
                print(f"  Error for {compound_id}: {e}")
            with id_to_cid_lock:
                id_to_cid[compound_id] = None
                completed_count[0] += 1
            return compound_id, None
    
    # Process compounds in batches with concurrent requests
    total_batches = (len(unique_ids) + batch_size - 1) // batch_size
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min((batch_num + 1) * batch_size, len(unique_ids))
            batch_ids = unique_ids[batch_start:batch_end]
            
            # Submit all requests in this batch
            future_to_id = {executor.submit(fetch_cid, cid): cid for cid in batch_ids}
            
            # Process completed requests with progress bar
            with tqdm(total=len(batch_ids), desc=f"Batch {batch_num + 1}/{total_batches}", leave=False) as pbar:
                for future in as_completed(future_to_id):
                    compound_id, cid = future.result()
                    pbar.update(1)
            
            # Update dataframe and save checkpoint
            df_result['PubChem CID'] = df_result['id'].map(id_to_cid)
            df_result.to_csv(output_csv_path, index=False)
            
            if batch_num < total_batches - 1:  # Don't delay after last batch
                time.sleep(delay)
    
    # Final update and save
    df_result['PubChem CID'] = df_result['id'].map(id_to_cid)
    df_result.to_csv(output_csv_path, index=False)
    
    print(f"\nDone! Results saved to {output_csv_path}")
    
    # Print summary
    total_rows = len(df_result)
    cids_found = df_result['PubChem CID'].notna().sum()
    print(f"\nSummary:")
    print(f"  Total rows: {total_rows}")
    print(f"  PubChem CIDs found: {cids_found} ({100*cids_found/total_rows:.1f}%)")
    print(f"  PubChem CIDs missing: {total_rows - cids_found} ({100*(total_rows-cids_found)/total_rows:.1f}%)")
    
    session.close()
    return df_result


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract PubChem CID from phytocompounds database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process the default phytocompounds_db.csv file
  python get_pubchem_cid_from_phytocompounds.py
  
  # Specify input and output files
  python get_pubchem_cid_from_phytocompounds.py -i input.csv -o output.csv
  
  # Adjust delay between requests (default: 0.5 seconds)
  python get_pubchem_cid_from_phytocompounds.py --delay 1.0
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        default='fart/dataset/individual-datasets/phytocompounds_db.csv',
        help='Input CSV file path (default: fart/dataset/individual-datasets/phytocompounds_db.csv)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output CSV file path (default: adds _with_cid to input name)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of compounds to process before saving checkpoint (default: 100)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=0.1,
        help='Delay between batches in seconds (default: 0.1)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=10,
        help='Number of concurrent threads for parallel requests (default: 10)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress for each compound'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        default=None,
        help='Test mode: test with a single compound ID (e.g., PMTDB00001)'
    )
    
    args = parser.parse_args()
    
    # Test mode: test with a single compound ID
    if args.test:
        print(f"Test mode: Fetching PubChem CID for {args.test}")
        cid = get_pubchem_cid_from_page(args.test)
        if cid:
            print(f"✓ Found PubChem CID: {cid}")
        else:
            print("✗ PubChem CID not found")
        return
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Process the file
    process_phytocompounds_csv(
        args.input,
        args.output,
        batch_size=args.batch_size,
        delay=args.delay,
        max_workers=args.max_workers,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()

