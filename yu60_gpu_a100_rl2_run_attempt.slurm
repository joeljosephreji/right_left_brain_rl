#!/bin/bash
# Usage: sbatch slurm-gpu-job-script
# Prepared By: Kai Xi,  Feb 2015
#              help@massive.org.au

#SBATCH --job-name=rl2_run
#SBATCH --account=yu60

# Request CPU resource for a serial job
#SBATCH --ntasks=12
#SBATCH --mem=32G

# Request for GPU, 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=3-00:00:00

# To receive an email when job completes or fails
#SBATCH --mail-user=jrej0002@student.monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the file for output (stdout)
#SBATCH --output=./slurm_logs/rl2_run-%j.out

# Set the file for error log (stderr)
#SBATCH --error=./slurm_logs/rl2_run-%j.err


# Command to run a gpu job
module load cuda
nvidia-smi
module load miniforge3
conda activate ../conda_envs/rl_env_39
python ./main.py  --env-type "ML3_rl2"

