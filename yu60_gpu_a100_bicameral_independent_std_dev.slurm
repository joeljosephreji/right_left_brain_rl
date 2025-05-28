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
python ./run_continual_learner.py --seed 808 --learning_rate 0.00001 --run_name "random_init_fixed20_gatepen_reach-v2" --env_name "reach-v2" --run_folder "rl2_baseline/rl2_bicameral_baseline" --num_mini_batch 8 --ppo_epoch 8 --num_processes 20 --randomization "random_init_fixed20" --algorithm "bicameral" --steps_per_env 5000000 --log_folder "logs/bicameral_net_independent_std_dev" --entropy_coef 0.00001 --use_gating_penalty True --gating_alpha 0.75 --gating_beta 5

