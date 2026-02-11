import json
import subprocess
import signal
import sys
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobProcess:
    """Tracks a single job process."""
    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.start_time = datetime.now()

class BatchManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.load_config()
        self.active_jobs = {}
        self.terminated = False

        # Handle shutdown signals
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def load_config(self):
        """Load main batch manager configuration."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info("BatchManager configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)

    def handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("Shutdown signal received. Terminating all jobs...")
        self.terminated = True
        self.stop_all_jobs()

    def start_processor(self):
        """Start the Processor script with its configuration."""
        processor_config_path = self.create_temp_processor_config()
        try:

            script_path = self.config.get("script", "processor_Chunks.py")
            if not os.path.exists(script_path):
                logger.error(f"Script '{script_path}' not found.")
                self.terminated = True
                return

            process = subprocess.Popen([
                sys.executable,
                script_path,
                '--config', processor_config_path
            ])
            self.active_jobs['processor'] = JobProcess(process)
            logger.info(f"Started Processor script with PID {process.pid}")
        except Exception as e:
            logger.error(f"Error starting Processor script: {e}")
            self.stop_all_jobs()

    def create_temp_processor_config(self):
        """Generate a temporary configuration for the Processor."""
        processor_config = {
            "input_file": self.config['input_file'],
            "output_file": self.config['output_file'],
            "endpoints": self.config['endpoints'],
            "num_jobs": self.config['num_jobs'],
            "model": self.config['model'],
            "amount_questions": self.config['amount_questions']
        }
        temp_config_path = "temp_processor_config.json"
        with open(temp_config_path, 'w') as f:
            json.dump(processor_config, f, indent=4)
        logger.info("Temporary Processor configuration file created.")
        return temp_config_path


    def monitor_jobs(self):
        """Monitor the active job process."""
        if 'processor' in self.active_jobs:
            process = self.active_jobs['processor'].process
            if process.poll() is not None:  # Process has finished
                logger.info("Processor script has completed.")
                del self.active_jobs['processor']

    def stop_all_jobs(self):
        """Stop all running processes."""
        for job_id in list(self.active_jobs.keys()):
            job = self.active_jobs[job_id]
            try:
                job.process.terminate()
                logger.info(f"Terminated job: {job_id}")
            except Exception as e:
                logger.error(f"Error stopping job {job_id}: {e}")
            del self.active_jobs[job_id]

    def run(self):
        """Run the BatchManager."""
        logger.info("BatchManager started.")
        self.start_processor()

        try:
            while not self.terminated and self.active_jobs:
                self.monitor_jobs()
                
        except Exception as e:
            logger.error(f"Error in BatchManager: {e}")
        finally:
            self.stop_all_jobs()
            logger.info("BatchManager shutdown complete.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python batch_manager.py <config_file>")
        sys.exit(1)

    config_path = sys.argv[1]
    manager = BatchManager(config_path)
    manager.run()

if __name__ == "__main__":
    main()
