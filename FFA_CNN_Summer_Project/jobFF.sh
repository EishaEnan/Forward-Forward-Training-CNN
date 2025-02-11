#! /usr/bin/env bash
#
#SBATCH --job-name=FFk3
#SBATCH --output=logsFF/MainFF_k3p1_%A_%a.log
#SBATCH --error=logsFF/ErrorFF_k3p1_%A_%a.log
#
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2         # Request 2 CPUs per task (total 8 CPUs: 4 tasks x 2 CPUs)
#SBATCH --time=10:00:00           # this sets the maximum time the job is allowed before killed
#SBATCH --partition=ampere24
#SBATCH --gres=gpu:a30:1

#SBATCH --array=1-10

echo "Starting job $SLURM_ARRAY_TASK_ID..."

source FF/bin/activate

python3.9 finalFF.py --run_id $SLURM_ARRAY_TASK_ID

echo "Job $SLURM_ARRAY_TASK_ID complete"

