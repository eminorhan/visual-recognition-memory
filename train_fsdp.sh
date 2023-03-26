#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=492GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train_gpt
#SBATCH --output=train_gpt_%A_%a.out

### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=16

LR=0.0003
OPTIMIZER='Adam'

# ALEF
#srun python -u /scratch/eo41/visual-recognition-memory/train.py \
#	--data_path '/scratch/work/public/imagenet/train' \
#	--save_dir '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models' \
#	--gpt_config 'GPT_alef' \
#	--save_prefix 'imagenet' \
#	--epochs 1000 \
#	--batch_size 128 \
#	--num_workers 8 \
#	--optimizer $OPTIMIZER \
#	--lr $LR \
#	--seed $SLURM_ARRAY_TASK_ID \
#	--resume '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models/imagenet_alef.pt'

# BET
#srun python -u /scratch/eo41/visual-recognition-memory/train.py \
#	--data_path '/scratch/work/public/imagenet/train' \
#	--save_dir '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models' \
#	--gpt_config 'GPT_bet' \
#	--save_prefix 'imagenet' \
#	--epochs 1000 \
#	--batch_size 64 \
#	--num_workers 8 \
#	--optimizer $OPTIMIZER \
#	--lr $LR \
#	--seed $SLURM_ARRAY_TASK_ID \
#	--resume '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models/imagenet_bet.pt'

# GIMEL
#srun python -u /scratch/eo41/visual-recognition-memory/train.py \
#	--data_path '/scratch/work/public/imagenet/train' \
#	--save_dir '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models' \
#	--gpt_config 'GPT_gimel' \
#	--save_prefix 'imagenet' \
#	--epochs 1000 \
#	--batch_size 32 \
#	--num_workers 8 \
#	--optimizer $OPTIMIZER \
#	--lr $LR \
#	--seed $SLURM_ARRAY_TASK_ID \
#	--resume '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models/imagenet_gimel.pt'

# DALET
srun python -u /scratch/eo41/visual-recognition-memory/train.py \
	--data_path '/scratch/work/public/imagenet/train' \
	--save_dir '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models' \
	--gpt_config 'GPT_dalet' \
	--save_prefix 'imagenet_dalet_fsdp' \
	--epochs 1000 \
	--batch_size 32 \
	--num_workers 16 \
	--optimizer $OPTIMIZER \
	--lr $LR \
	--seed $SLURM_ARRAY_TASK_ID \
	--resume ''
	
# GIMEL - SAYCam	
#srun python -u /scratch/eo41/visual-recognition-memory/train.py \
#	--data_path '/vast/eo41/SAY_1fps' \
#	--save_dir '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models' \
#	--gpt_config 'GPT_gimel' \
#	--save_prefix 'saycam' \
#	--epochs 1000 \
#	--batch_size 32 \
#	--num_workers 8 \
#	--optimizer $OPTIMIZER \
#	--lr $LR \
#	--seed $SLURM_ARRAY_TASK_ID \
#	--resume '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models/saycam_gimel.pt'

# GIMEL - ImageNet 10%
#srun python -u /scratch/eo41/visual-recognition-memory/train.py \
#	--data_path '/scratch/work/public/imagenet/train' \
#	--save_dir '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models' \
#	--gpt_config 'GPT_gimel' \
#	--save_prefix 'imagenet_10' \
#	--epochs 1000 \
#	--batch_size 32 \
#	--num_workers 8 \
#	--optimizer $OPTIMIZER \
#	--lr $LR \
#	--subsample 0.1 \
#	--seed $SLURM_ARRAY_TASK_ID \
#	--resume ''

# # GIMEL - ImageNet 1%
# srun python -u /scratch/eo41/visual-recognition-memory/train.py \
# 	--data_path '/scratch/work/public/imagenet/train' \
# 	--save_dir '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models' \
# 	--gpt_config 'GPT_gimel' \
# 	--save_prefix 'imagenet_1' \
# 	--epochs 1000 \
# 	--batch_size 32 \
# 	--num_workers 8 \
# 	--optimizer $OPTIMIZER \
# 	--lr $LR \
# 	--subsample 0.01 \
# 	--seed $SLURM_ARRAY_TASK_ID \
# 	--resume ''

echo "Done"
