#!/bin/bash
#SBATCH --job-name=train-cem           # Name of the job
#SBATCH --nodes=1                     # Request 1 nodes
#SBATCH --ntasks-per-node=1           # Start only 1 task per node, accelerate launches the different processes
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=10            # Request 10 cpus per task
#SBATCH --mem=60GB                    # Request 60Gb of cpu memory per node (should have 768 available)
#SBATCH --time=24:00:00               # Time limit hrs:min:sec (adjust as needed)
#SBATCH --output=slurm_logs/train_cem/output/train_%j.out  # Output file (%j will be replaced by job ID)
#SBATCH --error=slurm_logs/train_cem/error/train_%j.err   # Error log file
#SBATCH --partition=mcml-hgx-a100-80x4-mig
#SBATCH --qos=mcml


source /dss/dsshome1/04/go25kod3/miniforge3/etc/profile.d/conda.sh
conda activate CEM

cd concept-realignment-experiments

python train_base_models_and_save_predictions.py