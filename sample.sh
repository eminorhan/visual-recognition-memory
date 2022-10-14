#!/bin/bash

##SBATCH --account=cds
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=0:10:00
#SBATCH --array=0
#SBATCH --job-name=sample_gpt
#SBATCH --output=sample_gpt_%A_%a.out

module purge
module load cuda/11.3.1

python -u /scratch/eo41/visual-recognition-memory/sample.py \
	--gpt_dir '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models' \
	--condition 'cond' \
	--n_samples 6 \
	--seed 0 \
	--seed $SLURM_ARRAY_TASK_ID \
	--gpt_config 'GPT_gimel' \
	--gpt_model 'saycam_gimel' \
	--data_path '/scratch/eo41/data/cat.tar' # '/scratch/eo41/data/konkle_objects/konkle_objects_train_000000.tar'

echo "Done"
