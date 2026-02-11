import os
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Set, Dict
import logging
from urllib.parse import urlparse
import time
from datetime import datetime
import socket

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Worker:
    def __init__(self, input_dir: str, output_dir: str, domain_file: str, start_idx: int, end_idx: int):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.domain_file = Path(domain_file)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.tmp_dir = os.getenv('TMPDIR', '/tmp')
        self.domains = self._load_domains()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        self.hostname = socket.gethostname()
        self.worker_id = f"{self.hostname}_{self.start_idx}_{self.end_idx}"
        self.matched_patterns = {}  # Track frequency of matched patterns
        
    def _load_domains(self) -> Set[str]:
        with open(self.domain_file, 'r') as f:
            return {line.strip().lower() for line in f if line.strip()}

    def _get_domain_from_url(self, url: str) -> str:
        try:
            # Get the full netloc (domain) from the URL
            domain = urlparse(url).netloc.lower()
            # Remove common prefixes like www.
            domain = domain.replace('www.', '')
            return domain
        except:
            return ""

    def _is_domain_match(self, url_domain: str) -> bool:
        if not url_domain:
            return False
            
        # Split domain into parts (e.g., 'sub.example.com' -> ['sub', 'example', 'com'])
        domain_parts = url_domain.split('.')
        
        # Create all possible combinations of domain parts
        domain_combinations = []
        for i in range(len(domain_parts)):
            for j in range(i + 1, len(domain_parts) + 1):
                combination = '-'.join(domain_parts[i:j])
                domain_combinations.append(combination)
                # Also add version without hyphens
                if '-' in combination:
                    domain_combinations.append(combination.replace('-', ''))
        
        # Check if any of our domain patterns match
        for pattern in domain_combinations:
            if pattern in self.domains:
                # Track the matched pattern
                self.matched_patterns[pattern] = self.matched_patterns.get(pattern, 0) + 1
                return True
        return False

    def _get_all_jsonl_files(self) -> List[Path]:
        files = sorted([f for f in self.input_dir.glob('*.jsonl')])
        return files[self.start_idx:self.end_idx]

    def _get_processed_files(self) -> Set[str]:
        return {f.stem.replace('_filtered', '') for f in self.output_dir.glob('*_filtered.jsonl')}

    def _save_metrics(self, metrics: Dict) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.metrics_dir / f"metrics_{self.worker_id}_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def process_file(self, input_file: Path) -> None:
        if input_file.stem in self._get_processed_files():
            logging.info(f"Skipping {input_file.name} - already processed")
            return

        metrics = {
            "worker_id": self.worker_id,
            "file_name": input_file.name,
            "start_time": datetime.now().isoformat(),
            "input_size_bytes": input_file.stat().st_size,
            "processing_times": {},
            "matches": 0,
            "total_lines": 0,
            "errors": 0,
            "top_matched_patterns": {}  # Add to metrics
        }

        tmp_input = Path(self.tmp_dir) / input_file.name
        tmp_output = Path(self.tmp_dir) / f"{input_file.stem}_filtered.jsonl"
        final_output = self.output_dir / f"{input_file.stem}_filtered.jsonl"

        try:
            # Time copying to tmp
            start_time = time.time()
            logging.info(f"Copying {input_file.name} to temporary directory")
            shutil.copy2(input_file, tmp_input)
            metrics["processing_times"]["copy_to_tmp"] = time.time() - start_time

            # Time processing
            start_time = time.time()
            with open(tmp_input, 'r') as fin, open(tmp_output, 'w') as fout:
                for line in fin:
                    metrics["total_lines"] += 1
                    try:
                        data = json.loads(line)
                        source_url = data.get('source', '')
                        domain = self._get_domain_from_url(source_url)
                        
                        if self._is_domain_match(domain):
                            fout.write(line)
                            metrics["matches"] += 1
                    except json.JSONDecodeError:
                        metrics["errors"] += 1
                        continue
            metrics["processing_times"]["processing"] = time.time() - start_time

            # Time moving to final location
            start_time = time.time()
            logging.info(f"Moving filtered file to output directory: {final_output}")
            shutil.move(tmp_output, final_output)
            metrics["processing_times"]["move_to_output"] = time.time() - start_time

            # Calculate final metrics
            metrics["end_time"] = datetime.now().isoformat()
            metrics["output_size_bytes"] = final_output.stat().st_size
            metrics["match_rate"] = metrics["matches"] / metrics["total_lines"] if metrics["total_lines"] > 0 else 0
            metrics["total_duration"] = sum(metrics["processing_times"].values())
            metrics["processing_speed_mb_per_sec"] = (metrics["input_size_bytes"] / 1024 / 1024) / metrics["total_duration"]
            
            # Add top 100 matched patterns to metrics
            top_patterns = sorted(self.matched_patterns.items(), key=lambda x: x[1], reverse=True)[:100]
            metrics["top_matched_patterns"] = dict(top_patterns)
            
            # Save metrics
            self._save_metrics(metrics)
            
            logging.info(f"Processing complete for {input_file.name}:")
            logging.info(f"Total duration: {metrics['total_duration']:.2f} seconds")
            logging.info(f"Processing speed: {metrics['processing_speed_mb_per_sec']:.2f} MB/s")
            logging.info(f"Match rate: {metrics['match_rate']*100:.2f}%")
            logging.info("Top 100 matched patterns:")
            for pattern, count in top_patterns:
                logging.info(f"  {pattern}: {count} matches")
        finally:
            # Cleanup temporary files
            if tmp_input.exists():
                tmp_input.unlink()
            if tmp_output.exists():
                tmp_output.unlink()

    def run(self):
        start_time = time.time()
        files_to_process = self._get_all_jsonl_files()
        logging.info(f"Processing files {self.start_idx} to {self.end_idx}")
        
        overall_metrics = {
            "worker_id": self.worker_id,
            "start_time": datetime.now().isoformat(),
            "files_processed": 0,
            "total_input_size_bytes": 0,
            "total_output_size_bytes": 0,
            "total_matches": 0,
            "total_lines": 0,
            "total_errors": 0,
            "files": [],
            "overall_top_matched_patterns": {}  # Add to overall metrics
        }

        for file in files_to_process:
            try:
                logging.info(f"Processing file: {file.name}")
                metrics_file = list(self.metrics_dir.glob(f"metrics_{self.worker_id}*_{file.stem}.json"))
                if metrics_file:
                    with open(metrics_file[0]) as f:
                        file_metrics = json.load(f)
                        overall_metrics["files"].append(file_metrics)
                        overall_metrics["files_processed"] += 1
                        overall_metrics["total_input_size_bytes"] += file_metrics["input_size_bytes"]
                        overall_metrics["total_output_size_bytes"] += file_metrics["output_size_bytes"]
                        overall_metrics["total_matches"] += file_metrics["matches"]
                        overall_metrics["total_lines"] += file_metrics["total_lines"]
                        overall_metrics["total_errors"] += file_metrics["errors"]
                
                self.process_file(file)
            except Exception as e:
                logging.error(f"Error processing {file.name}: {str(e)}")

        overall_metrics["end_time"] = datetime.now().isoformat()
        overall_metrics["total_duration"] = time.time() - start_time
        overall_metrics["avg_processing_speed_mb_per_sec"] = (
            overall_metrics["total_input_size_bytes"] / 1024 / 1024
        ) / overall_metrics["total_duration"]
        
        # Add overall top 100 matched patterns
        top_patterns = sorted(self.matched_patterns.items(), key=lambda x: x[1], reverse=True)[:100]
        overall_metrics["overall_top_matched_patterns"] = dict(top_patterns)
        
        logging.info("Overall Top 100 matched patterns:")
        for pattern, count in top_patterns:
            logging.info(f"  {pattern}: {count} matches")

        self._save_metrics(overall_metrics)

def main():
    parser = argparse.ArgumentParser(description='Process JSONL files with domain filtering')
    parser.add_argument('--input-dir', required=True, help='Input directory containing JSONL files')
    parser.add_argument('--output-dir', required=True, help='Output directory for filtered files')
    parser.add_argument('--domain-file', required=True, help='File containing domain names to filter')
    parser.add_argument('--start-idx', type=int, required=True, help='Start index of files to process')
    parser.add_argument('--end-idx', type=int, required=True, help='End index of files to process')

    args = parser.parse_args()

    worker = Worker(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        domain_file=args.domain_file,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    worker.run()

if __name__ == '__main__':
    main()