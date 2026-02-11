#!/usr/bin/env python3
"""
Data Chunker - Splits large JSON files into manageable chunks for processing.
Handles streaming JSON parsing to avoid memory issues with large files.
"""
import json
import os
import argparse
from datetime import datetime
import yaml

try:
    import ijson
except ImportError:
    print("Installing required package: ijson")
    os.system('pip install ijson')
    import ijson


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ChunkWriter:
    """Handles writing JSON data in chunks to separate files."""
    
    def __init__(self, output_dir: str, chunk_size_bytes: int):
        self.output_dir = output_dir
        self.chunk_size_bytes = chunk_size_bytes
        self.current_chunk = 1
        self.current_size = 0
        self.current_file = None
        self.items_written = 0
        self.ensure_output_dir()
        self.start_new_chunk()

    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)

    def start_new_chunk(self):
        """Start a new chunk file."""
        if self.current_file:
            self.end_chunk()
        filename = os.path.join(self.output_dir, f'chunk{self.current_chunk}.json')
        self.current_file = open(filename, 'w', buffering=1024*1024, encoding='utf-8')
        self.current_file.write('[\n')
        self.current_size = 2
        print(f"Starting chunk{self.current_chunk}.json")

    def write_item(self, item: dict):
        """Write a single item to the current chunk."""
        json_str = json.dumps(item, ensure_ascii=False)
        item_size = len(json_str.encode('utf-8')) + 2

        if self.current_size + item_size > self.chunk_size_bytes and self.current_size > 2:
            self.current_chunk += 1
            self.start_new_chunk()

        if self.current_size > 2:
            self.current_file.write(',\n')
        self.current_file.write(json_str)
        self.current_size += item_size
        self.items_written += 1

    def end_chunk(self):
        """Close the current chunk file."""
        if self.current_file:
            self.current_file.write('\n]')
            self.current_file.close()
            self.current_file = None


def process_json_file(input_path: str, chunk_writer: ChunkWriter):
    """Process a single JSON file using streaming parser."""
    try:
        with open(input_path, 'rb') as file:
            parser = ijson.parse(file)
            current_item = {}
            current_key = None
            array_depth = 0
            
            for prefix, event, value in parser:
                if prefix == '' and event == 'start_array':
                    array_depth += 1
                elif prefix == '' and event == 'end_array':
                    array_depth -= 1
                elif array_depth == 1:
                    if event == 'start_map':
                        current_item = {}
                    elif event == 'end_map':
                        chunk_writer.write_item(current_item)
                        current_item = {}
                    elif event == 'map_key':
                        current_key = value
                    elif current_key:
                        current_item[current_key] = value
                        current_key = None
                        
    except ijson.JSONError as e:
        print(f"Error parsing {input_path}: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error processing {input_path}: {str(e)}")
        return False
    return True


def process_all_files(input_dir: str, output_dir: str, chunk_size_mb: int):
    """Process all JSON files in the input directory."""
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])
    
    if not input_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    total_input_size = sum(os.path.getsize(os.path.join(input_dir, f)) for f in input_files)
    
    print(f"Found {len(input_files)} JSON files to process")
    print(f"Total input size: {total_input_size / (1024*1024*1024):.2f} GB")
    
    chunk_writer = ChunkWriter(output_dir, chunk_size_bytes)
    
    for filename in input_files:
        input_path = os.path.join(input_dir, filename)
        file_size = os.path.getsize(input_path)
        print(f"\nProcessing {filename} ({file_size / (1024*1024):.2f} MB)")
        
        if not process_json_file(input_path, chunk_writer):
            print(f"Failed to process {filename}")
            continue
    
    chunk_writer.end_chunk()
    
    total_output_size = sum(
        os.path.getsize(os.path.join(output_dir, f)) 
        for f in os.listdir(output_dir) if f.endswith('.json')
    )
    
    print("\n=== Processing Statistics ===")
    print(f"Total input size: {total_input_size / (1024*1024*1024):.2f} GB")
    print(f"Total output size: {total_output_size / (1024*1024*1024):.2f} GB")
    print(f"Total items written: {chunk_writer.items_written:,}")
    print(f"Number of chunks created: {chunk_writer.current_chunk}")


def main():
    parser = argparse.ArgumentParser(
        description="Split large JSON files into manageable chunks"
    )
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing input JSON files'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory for output chunk files'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--chunk-size-mb',
        type=int,
        help='Override chunk size from config (in MB)'
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    chunk_size_mb = args.chunk_size_mb or config['data']['chunk_size_mb']
    
    start_time = datetime.now()
    print(f"Starting JSON processing at {start_time}")
    
    process_all_files(args.input_dir, args.output_dir, chunk_size_mb)
    
    end_time = datetime.now()
    print(f"\nProcessing completed at {end_time}")
    print(f"Total time: {end_time - start_time}")


if __name__ == "__main__":
    main()
