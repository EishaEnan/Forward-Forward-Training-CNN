#! /usr/bin/env bash
#
#SBATCH --job-name=finalFF
#SBATCH --output=logsFF/main_%A_%a.log
#SBATCH --error=logsFF/error_%A_%a.log
#
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2         # Request 2 CPUs per task (total 8 CPUs: 4 tasks x 2 CPUs)
#SBATCH --time=10:00:00           # this sets the maximum time the job is allowed before killed
#SBATCH --partition=ampere24
#SBATCH --gres=gpu:a30:1

#SBATCH --array=1-10

##SBATCH --partition=cpu # the double hash means that SLURM won't read this line.

# load the python module

echo "Starting job $SLURM_ARRAY_TASK_ID..."

source FF/bin/activate

python3.9 finalFF.py --run_id $SLURM_ARRAY_TASK_ID



echo "Job $SLURM_ARRAY_TASK_ID complete"

