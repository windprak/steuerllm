#!/usr/bin/env python3
"""
Orchestrator - Distributes conversation generation work across multiple worker endpoints.
Useful for distributed processing with multiple GPUs or machines.
"""
import subprocess
import os
import glob
import argparse
from itertools import cycle
from typing import List, Tuple
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_input_chunks(input_dir: str, pattern: str = "*.json") -> List[str]:
    """Get all JSON files from input directory."""
    chunk_files = glob.glob(os.path.join(input_dir, pattern))
    chunk_files = sorted(chunk_files)
    return chunk_files


def ensure_dirs(output_dir: str, num_workers: int):
    """Ensure necessary directories exist."""
    os.makedirs("logs", exist_ok=True)
    for i in range(1, num_workers + 1):
        os.makedirs(os.path.join(output_dir, f"worker_{i}"), exist_ok=True)


def chunk_list(lst: List[str], n: int) -> List[List[str]]:
    """Split list into chunks of size n."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def submit_work(
    endpoint: str, 
    chunk_files: List[str], 
    worker_idx: int, 
    output_base: str,
    config: dict
) -> subprocess.Popen:
    """Submit work to a specific endpoint with given chunk files."""
    input_files = ",".join(chunk_files)
    output_dir = os.path.join(output_base, f"worker_{worker_idx}")
    log_base = f"worker_{worker_idx}"
    
    cmd = [
        "python", "src/worker.py",
        "--input-files", input_files,
        "--output-dir", output_dir,
        "--config", config['config_path'],
        "--max-concurrent", str(config['worker']['max_concurrent'])
    ]
    
    print(f"\nSubmitting to {endpoint}:")
    print(f"  Worker ID: {worker_idx}")
    print(f"  Input files: {len(chunk_files)} chunks")
    print(f"  Output dir: {output_dir}")
    print(f"  Logs: logs/{log_base}_out.log, logs/{log_base}_err.log")
    
    with open(f"logs/{log_base}_out.log", 'w') as out_f, \
         open(f"logs/{log_base}_err.log", 'w') as err_f:
        
        env = os.environ.copy()
        env['API_ENDPOINT'] = endpoint
        
        process = subprocess.Popen(
            cmd,
            stdout=out_f,
            stderr=err_f,
            env=env
        )
        return process


def get_worker_status(output_base: str, worker_idx: int) -> float:
    """Get the latest modification time of output files for a worker."""
    output_dir = os.path.join(output_base, f"worker_{worker_idx}")
    if not os.path.exists(output_dir):
        return 0
        
    jsonl_files = glob.glob(os.path.join(output_dir, "*.jsonl"))
    if not jsonl_files:
        return 0
        
    return max(os.path.getmtime(f) for f in jsonl_files)


def create_work_assignments(
    chunk_groups: List[List[str]], 
    endpoints: List[str], 
    workers_per_endpoint: int
) -> List[Tuple[str, List[str]]]:
    """Create work assignments distributing chunks across endpoints."""
    all_slots = []
    for endpoint in endpoints:
        all_slots.extend([endpoint] * workers_per_endpoint)
    
    assignments = []
    for chunks, endpoint in zip(chunk_groups, cycle(all_slots)):
        if chunks:
            assignments.append((endpoint, chunks))
    
    return assignments


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate distributed conversation generation"
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--input-dir',
        help='Override input directory from config'
    )
    parser.add_argument(
        '--output-dir',
        help='Override output directory from config'
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    config['config_path'] = args.config
    
    if not config['orchestrator']['enabled']:
        print("Orchestrator is not enabled in config. Set orchestrator.enabled to true.")
        return
    
    if not config['orchestrator']['endpoints']:
        print("No endpoints configured. Add endpoints to config.yaml")
        return
    
    input_dir = args.input_dir or config['data']['input_dir']
    output_dir = args.output_dir or config['data']['output_dir']
    
    endpoints = config['orchestrator']['endpoints']
    workers_per_endpoint = config['orchestrator']['workers_per_endpoint']
    chunks_per_worker = config['orchestrator']['chunks_per_worker']
    
    print(f"Found {len(endpoints)} endpoints")
    print(f"Workers per endpoint: {workers_per_endpoint}")
    print(f"Chunks per worker: {chunks_per_worker}")
    
    ensure_dirs(output_dir, len(endpoints) * workers_per_endpoint)
    
    all_chunks = get_input_chunks(input_dir)
    print(f"\nFound {len(all_chunks)} input chunks")
    
    if not all_chunks:
        print("No input chunks found. Run data_chunker.py first.")
        return
    
    chunk_groups = chunk_list(all_chunks, chunks_per_worker)
    print(f"Created {len(chunk_groups)} work groups")
    
    work_assignments = create_work_assignments(
        chunk_groups, 
        endpoints, 
        workers_per_endpoint
    )
    print(f"\nCreated {len(work_assignments)} work assignments")
    
    processes = []
    for idx, (endpoint, chunks) in enumerate(work_assignments, 1):
        process = submit_work(endpoint, chunks, idx, output_dir, config)
        processes.append(process)
    
    print(f"\n{'='*60}")
    print(f"Launched {len(processes)} workers")
    print(f"Monitor progress in logs/ directory")
    print(f"{'='*60}")
    
    print("\nWaiting for all workers to complete...")
    for p in processes:
        p.wait()
    
    print("\n✓ All workers completed")


if __name__ == "__main__":
    main()
