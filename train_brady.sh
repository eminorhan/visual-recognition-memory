#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=00:15:00
#SBATCH --array=0
#SBATCH --job-name=train_gpt
#SBATCH --output=train_gpt_%A_%a.out

### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

module purge
module load cuda/11.3.1

LR=0.0001
OPTIMIZER='Adam'

srun python -u /scratch/eo41/visual-recognition-memory/train.py \
	--data_path '/scratch/eo41/brady_1/study' \
	--save_dir '/vast/eo41/visual-recognition-memory-models/' \
	--gpt_config 'GPT_dalet' \
	--save_prefix 'brady_1_imagenet' \
	--epochs 5 \
	--batch_size 1 \
	--num_workers 8 \
	--optimizer $OPTIMIZER \
	--lr $LR \
	--seed $SLURM_ARRAY_TASK_ID \
	--resume '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models/imagenet_dalet.pt'

#srun python -u /scratch/eo41/visual-recognition-memory/train.py \
#	--data_path '/scratch/eo41/brady_2/study' \
#	--save_dir '/vast/eo41/visual-recognition-memory-models/' \
#	--gpt_config 'GPT_gimel' \
#	--save_prefix 'brady_2_imagenet' \
#	--epochs 50 \
#	--batch_size 45 \
#	--num_workers 8 \
#	--optimizer $OPTIMIZER \
#	--lr $LR \
#	--seed $SLURM_ARRAY_TASK_ID \
#	--resume '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models/imagenet_gimel.pt'
	
echo "Done"
