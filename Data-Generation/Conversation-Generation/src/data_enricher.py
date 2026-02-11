#!/usr/bin/env python3
"""
Data Enricher - Enriches exported conversation data with original metadata.
Matches generated conversations with source documents to create complete records.
"""
import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_json_array_streaming(file_path: Path):
    """Parse a large JSON array file by accumulating complete objects."""
    print(f"  Parsing {file_path.name}...")
    records = []
    buffer = ""
    brace_count = 0
    in_object = False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            
            if stripped == '[':
                continue
            
            if stripped == ']':
                break
            
            if not stripped:
                continue
            
            buffer += line
            
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_object = True
                elif char == '}':
                    brace_count -= 1
            
            if in_object and brace_count == 0:
                obj_str = buffer.strip()
                if obj_str.endswith(','):
                    obj_str = obj_str[:-1]
                
                try:
                    record = json.loads(obj_str)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"    Warning: Failed to parse object at position {len(records)}: {str(e)[:100]}")
                
                buffer = ""
                in_object = False
                brace_count = 0
    
    print(f"  Parsed {len(records)} records from {file_path.name}")
    return records


def load_source_data(source_dir: Path):
    """Load all source JSON files into a lookup dictionary."""
    print("Loading source data...")
    lookup = {}
    
    source_files = sorted(source_dir.glob("*.json"))
    
    if not source_files:
        print(f"Warning: No source files found in {source_dir}")
        return lookup
    
    for source_file in source_files:
        records = parse_json_array_streaming(source_file)
        
        for record in records:
            record_id = record.get("_recordid")
            if record_id:
                enriched_record = {k: v for k, v in record.items() if k != "Doc"}
                lookup[record_id] = enriched_record
    
    print(f"\nTotal records loaded: {len(lookup)}")
    return lookup


def process_file(args):
    """Process one conversation JSONL file and enrich with source data."""
    conversation_file, lookup = args
    print(f"Processing {conversation_file.name}...")
    enriched = []
    not_found = 0
    
    with open(conversation_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                conv = json.loads(line)
                rid = conv.get("_recordid")
                
                if rid in lookup:
                    rec = lookup[rid].copy()
                    rec["conversation"] = conv.get("conversation", [])
                    enriched.append(rec)
                else:
                    enriched.append(conv)
                    not_found += 1
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse line in {conversation_file.name}: {str(e)[:100]}")
    
    print(f"Done {conversation_file.name}: {len(enriched)} records ({not_found} not matched)")
    return enriched


def write_output(records: list, output_file: Path):
    """Write enriched records to JSONL file."""
    print(f"\nWriting {len(records)} records to {output_file.name}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    file_size = output_file.stat().st_size / (1024 * 1024)
    print(f"  Written: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Enrich conversation data with original metadata"
    )
    parser.add_argument(
        '--source-dir',
        required=True,
        help='Directory containing original source JSON files'
    )
    parser.add_argument(
        '--conversation-dir',
        required=True,
        help='Directory containing generated conversation JSONL files'
    )
    parser.add_argument(
        '--output-file',
        required=True,
        help='Output JSONL file for enriched data'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=cpu_count(),
        help='Number of parallel workers'
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    conversation_dir = Path(args.conversation_dir)
    output_file = Path(args.output_file)
    
    print("=" * 80)
    print("Enriching Conversation Data with Original Metadata")
    print("=" * 80)
    
    lookup = load_source_data(source_dir)
    
    conversation_files = sorted(conversation_dir.glob("*.jsonl"))
    print(f"\nFound {len(conversation_files)} conversation files to process")
    
    if not conversation_files:
        print(f"No conversation files found in {conversation_dir}")
        return
    
    print(f"\nProcessing files with {args.workers} workers...")
    
    with Pool(args.workers) as pool:
        results = pool.map(process_file, [(f, lookup) for f in conversation_files])
    
    print("\nCombining results...")
    all_records = []
    for r in results:
        all_records.extend(r)
    
    print(f"Total enriched records: {len(all_records)}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_output(all_records, output_file)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()
