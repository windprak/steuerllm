#!/usr/bin/env python3

import argparse
from pathlib import Path
import logging
import pandas as pd
from typing import Set, Tuple
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_urls(input_file: str) -> Set[str]:
    """
    Load URLs from file into a set for O(1) lookup time.
    
    Args:
        input_file: Path to input URLs file
        
    Returns:
        Set of URLs
    """
    with open(input_file, 'r') as f:
        return {line.strip() for line in f}

def process_parquet_batch(parquet_file: str, urls_set: Set[str], batch_size: int = 100000) -> Tuple[int, int]:
    """
    Process parquet file in batches and match URLs.
    
    Args:
        parquet_file: Path to parquet file
        urls_set: Set of URLs to match against
        batch_size: Number of rows to process at once
        
    Returns:
        Tuple of (total_rows, matched_rows)
    """
    total_rows = 0
    matched_rows = 0
    
    # Open the parquet file
    parquet_table = pq.read_table(parquet_file)
    num_rows = parquet_table.num_rows
    
    # Process in batches
    for batch_start in range(0, num_rows, batch_size):
        batch_end = min(batch_start + batch_size, num_rows)
        batch = parquet_table.slice(batch_start, batch_end - batch_start).to_pandas()
        
        # Perform URL matching on the batch
        matches = batch['url'].isin(urls_set)
        matched_rows += matches.sum()
        total_rows += len(batch)
        
        if batch_start % (batch_size * 10) == 0:
            logging.info(f"Processed {total_rows:,} rows, found {matched_rows:,} matches...")
    
    return total_rows, matched_rows

def filter_domain_names(input_file: str, parquet_file: str, output_file: str) -> Tuple[int, int]:
    """
    Filter parquet file based on URL matches from domain names file.
    
    Args:
        input_file: Path to input URLs file
        parquet_file: Path to parquet file to filter
        output_file: Path to write filtered parquet file
        
    Returns:
        Tuple of (total_count, matched_count)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load URLs into set for O(1) lookup
    logging.info("Loading URLs into memory...")
    urls_set = load_urls(input_file)
    logging.info(f"Loaded {len(urls_set):,} URLs")
    
    # Process the parquet file
    logging.info("Processing parquet file...")
    total_count, matched_count = process_parquet_batch(parquet_file, urls_set)
    
    # Create filtered parquet file
    logging.info("Creating filtered parquet file...")
    df = pd.read_parquet(parquet_file)
    filtered_df = df[df['url'].isin(urls_set)]
    filtered_df.to_parquet(output_file)
    
    logging.info(f"Total rows processed: {total_count:,}")
    logging.info(f"Matched rows: {matched_count:,}")
    logging.info(f"Output written to: {output_file}")
    
    return total_count, matched_count

def main():
    parser = argparse.ArgumentParser(description='Filter parquet file based on URL matches')
    parser.add_argument('input_file', help='Input URLs file path')
    parser.add_argument('parquet_file', help='Input parquet file path')
    parser.add_argument('output_file', help='Output filtered parquet file path')
    args = parser.parse_args()
    
    filter_domain_names(args.input_file, args.parquet_file, args.output_file)

if __name__ == '__main__':
    main()
