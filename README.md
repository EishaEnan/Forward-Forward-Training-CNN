# Forward-Forward (FF) and Backpropagation (BP) Training

This repository contains scripts for training neural networks using both Forward-Forward (FF) and Backpropagation (BP) methods. It also includes tools for analyzing training logs and aggregating results.

## Repository Structure
- `finalBP.py`: Script for training a neural network using **Backpropagation (BP)**.
- `finalFF.py`: Script for training a neural network using **Forward-Forward (FF)**.
- `analyze.py`: Analyzes log files from training runs and generates aggregated results.
- `jobBP.sh`: Shell script for array job submission of BP training.
- `jobFF.sh`: Shell script for array job submission of FF training.
- `logsBP/`: Stores training logs for BP runs.
- `jobTemplate.sh`: Template shell script for single job submission.
- `logsFF/`: Stores training logs for FF runs.
- `data/`: Contains datasets used for training.
- `InputImage/`: Stores a few input and label-encoded images.
- `FF/`: Virtual environment for dependencies.
- `aggregated_results/`: Stores CSV and PNG files generated from `analyze.py`.
- `requirements.txt`: List of dependencies required to run the scripts.

## Setup
1. Clone the repository:
   ```sh
   git clone <repo_url>
   cd <repo_name>
   ```
2. Create and activate the virtual environment:
   ```sh
   python3 -m venv FF
   source FF/bin/activate  # On macOS/Linux
   FF\Scripts\activate     # On Windows
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Training
### Backpropagation (BP)
To train a model using **BP**, in your jobBP.sh include the command:
```sh
python3 finalBP.py
```

### Forward-Forward (FF)
To train a model using **FF**, in your jobFF.sh include the command:
```sh
python3 finalFF.py
```

## Submitting Jobs to HPC
To submit a job to the HPC using Slurm, run:
```sh
sbatch jobBP.sh  # For Backpropagation training
sbatch jobFF.sh  # For Forward-Forward training
```
For a single non-array job, use the following template:
```sh
sbatch <single_job_script>.sh
```
Ensure the job script specifies the required parameters and resources.

## Log Analysis
To analyze log files and generate results, use:
```sh
python3 analyze.py "logsBP/<log_filename_pattern_*>"
python3 analyze.py "logsFF/<log_filename_pattern_*>"
```
The results will be saved in `aggregated_results/` as CSV and PNG files.

## License
MIT License.


