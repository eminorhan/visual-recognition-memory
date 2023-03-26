#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=finetune_gpt
#SBATCH --output=finetune_gpt_%A_%a.out

### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

module purge
module load cuda/11.3.1

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

# DALET
#srun python -u /scratch/eo41/visual-recognition-memory/train.py \
#	--data_path '/scratch/work/public/imagenet/train' \
#	--save_dir '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models' \
#	--gpt_config 'GPT_dalet' \
#	--save_prefix 'imagenet' \
#	--epochs 1000 \
#	--batch_size 16 \
#	--num_workers 8 \
#	--optimizer $OPTIMIZER \
#	--lr $LR \
#	--seed $SLURM_ARRAY_TASK_ID \
#	--resume '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models/imagenet_dalet.pt'
	
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

# # GIMEL - ImageNet 100%
# srun python -u /scratch/eo41/visual-recognition-memory/train.py \
# 	--data_path '/vast/eo41/data/konkle_ood/vehicle_vs_nonvehicle/nonvehicle' \
# 	--save_dir '/scratch/eo41/visual-recognition-memory/gpt_finetuned_models' \
# 	--gpt_config 'GPT_gimel' \
# 	--save_prefix 'imagenet100_gimel_konkle_nonvehicle' \
# 	--save_freq 100 \
# 	--epochs 1000 \
# 	--batch_size 32 \
# 	--num_workers 16 \
# 	--optimizer $OPTIMIZER \
# 	--lr $LR \
# 	--seed $SLURM_ARRAY_TASK_ID \
# 	--resume '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models/imagenet_gimel.pt'

# # GIMEL - ImageNet 10%
# srun python -u /scratch/eo41/visual-recognition-memory/train.py \
# 	--data_path '/vast/eo41/data/konkle_ood/vehicle_vs_nonvehicle/nonvehicle' \
# 	--save_dir '/scratch/eo41/visual-recognition-memory/gpt_finetuned_models' \
# 	--gpt_config 'GPT_gimel' \
# 	--save_prefix 'imagenet10_gimel_konkle_nonvehicle' \
# 	--save_freq 100 \
# 	--epochs 1000 \
# 	--batch_size 32 \
# 	--num_workers 16 \
# 	--optimizer $OPTIMIZER \
# 	--lr $LR \
# 	--seed $SLURM_ARRAY_TASK_ID \
# 	--resume '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models/imagenet_10_gimel.pt'

# GIMEL - ImageNet 1%
srun python -u /scratch/eo41/visual-recognition-memory/train.py \
	--data_path '/vast/eo41/data/konkle_ood/vehicle_vs_nonvehicle/nonvehicle' \
	--save_dir '/scratch/eo41/visual-recognition-memory/gpt_finetuned_models' \
	--gpt_config 'GPT_gimel' \
	--save_prefix 'imagenet1_gimel_konkle_nonvehicle' \
	--save_freq 100 \
	--epochs 1000 \
	--batch_size 32 \
	--num_workers 16 \
	--optimizer $OPTIMIZER \
	--lr $LR \
	--seed $SLURM_ARRAY_TASK_ID \
	--resume '/scratch/eo41/visual-recognition-memory/gpt_pretrained_models/imagenet_1_gimel.pt'

echo "Done"
