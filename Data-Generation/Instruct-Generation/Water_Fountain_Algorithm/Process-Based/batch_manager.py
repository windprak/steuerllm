import json
import subprocess
import os
import sys
import time
import logging
import psutil
from datetime import datetime
from typing import Dict, List
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_manager.log'),
        logging.StreamHandler()
    ]
)

class JobProcess:
    def __init__(self, job_config: Dict, process: subprocess.Popen):
        self.job_config = job_config
        self.process = process
        self.start_time = datetime.now()

    def check_resource_limits(self) -> bool:
        """Check if the process is within its resource limits."""
        try:
            process = psutil.Process(self.process.pid)
            
            # Check memory usage
            memory_usage_gb = process.memory_info().rss / (1024 * 1024 * 1024)
            max_memory_gb = float(self.job_config['resource_limits']['max_memory_gb'])
            if memory_usage_gb > max_memory_gb:
                logging.warning(f"Job {self.job_config['job_id']} exceeded memory limit: {memory_usage_gb:.2f}GB")
                return False

            # Check CPU usage
            cpu_percent = process.cpu_percent()
            max_cpu_percent = float(self.job_config['resource_limits']['max_cpu_percent'])
            if cpu_percent > max_cpu_percent:
                logging.warning(f"Job {self.job_config['job_id']} exceeded CPU limit: {cpu_percent}%")
                return False

            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

class BatchManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.load_config()
        self.active_jobs: Dict[str, JobProcess] = {}
        self.terminated = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logging.info("Configuration loaded successfully")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            sys.exit(1)

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logging.info("Shutdown signal received. Terminating all jobs...")
        self.terminated = True
        self.stop_all_jobs()

    def start_all_jobs(self) -> None:
        """Start all enabled jobs."""
        try:
            global_config = self.config.get('global_settings', {})
            script_path = global_config.get('script', 'hybrid_search_Diversity.py')
            
            if not os.path.exists(script_path):
                logging.error(f"Script file '{script_path}' not found.")
                return
            
            # Start a single process that will handle all jobs
            process = subprocess.Popen([
                sys.executable,
                script_path,
                '--config', self.config_path  # Use the main config file directly
            ])
            
            # Store as a single job process
            self.active_jobs['main'] = JobProcess(self.config['jobs'][0], process)  # Use first job's config for resource limits
            logging.info(f"Started batch processing with PID {process.pid}")
            
        except Exception as e:
            logging.error(f"Error in batch manager: {e}")
            self.stop_all_jobs()

    def stop_job(self, job_id: str):
        """Stop a specific job."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            try:
                # Clean up the temporary config file
                temp_config = f"temp_config_{job_id}.json"
                if os.path.exists(temp_config):
                    os.remove(temp_config)
                
                # Terminate the process
                job.process.terminate()
                logging.info(f"Terminated job {job_id}")
            except Exception as e:
                logging.error(f"Error stopping job {job_id}: {e}")
            del self.active_jobs[job_id]

    def stop_all_jobs(self):
        """Stop all active jobs."""
        for job_id in list(self.active_jobs.keys()):
            self.stop_job(job_id)

    def monitor_jobs(self):
        """Monitor active jobs and their resource usage."""
        for job_id, job_process in list(self.active_jobs.items()):
            # Check if process is still running
            if job_process.process.poll() is not None:
                logging.info(f"Job {job_id} has finished with return code {job_process.process.returncode}")
                del self.active_jobs[job_id]
                continue

            # Check resource limits
            if not job_process.check_resource_limits():
                logging.warning(f"Job {job_id} exceeded resource limits. Terminating...")
                self.stop_job(job_id)

    def run(self):
        """Main execution loop for the batch manager."""
        logging.info("Starting batch manager...")
        
        try:
            # Start enabled jobs
            self.start_all_jobs()

            # Monitor jobs until all are complete or termination is requested
            while not self.terminated and self.active_jobs:
                self.monitor_jobs()
                time.sleep(10)  # Check status every 10 seconds

        except Exception as e:
            logging.error(f"Error in batch manager: {e}")
        finally:
            self.stop_all_jobs()
            logging.info("Batch manager shutdown complete")

def main():
    if len(sys.argv) != 2:
        print("Usage: python batch_manager.py <config_file>")
        sys.exit(1)

    config_path = sys.argv[1]
    manager = BatchManager(config_path)
    manager.run()

if __name__ == "__main__":
    main()
