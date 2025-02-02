#! /usr/bin/env bash
#
#SBATCH --job-name=finalFF
#SBATCH --output=logsFF/main%j.log
#SBATCH --error=logsFF/error%j.log
#
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2         # Request 2 CPUs per task (total 8 CPUs: 4 tasks x 2 CPUs)
#SBATCH --time=10:00:00           # this sets the maximum time the job is allowed before killed
#SBATCH --partition=ampere24
#SBATCH --gres=gpu:a30:1

##SBATCH --partition=cpu # the double hash means that SLURM won't read this line.

# load the python module
# module load PyTorch/Python3.10 # make sure to load the modules needed
echo "Job Starting..."

source FF/bin/activate
# echo "venv activated"

python3.9 finalFF.py



echo "job complete"

