#! /usr/bin/env bash
#
#SBATCH --job-name=<JOB_NAME> # Replace with an appropriate job name
#SBATCH --output=<logs_path>/<JOB_NAME>_%A.log  # Log file
#SBATCH --error=<logs_path>/<JOB_NAME>_error_%A.log  # Error log file
#
#SBATCH --ntasks=1  # Number of tasks (single job)
#SBATCH --cpus-per-task=2  # Number of CPUs per task
#SBATCH --time=10:00:00  # Maximum execution time
#SBATCH --partition=<PARTITION_NAME>  # Specify partition
#SBATCH --gres=gpu:<GPU_TYPE>:1  # Adjust GPU type if needed

echo "Starting job..."

source FF/bin/activate  # Activate virtual environment

python3.9 <SCRIPT_NAME>.py

echo "Job complete"

