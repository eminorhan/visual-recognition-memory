#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240GB
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

EXS=("brady_1" "brady_2")
LRS=(0.0001 0.00003 0.00001)
BSS=(1 4 16)
SDS=(0 1)

# ALEF
for EX in "${EXS[@]}"
do
	for LR in "${LRS[@]}"
	do
		for BS in "${BSS[@]}"
		do
			for SD in "${SDS[@]}"
			do
				SP="imagenet_alef_${EX}_${LR}_${BS}_${SD}"
				srun python -u /scratch/eo41/visual-recognition-memory/train.py \
					--data_path "/scratch/eo41/visual-recongition-memory-datasets/${EX}/study" \
					--save_dir "/vast/eo41/visual-recognition-memory-models/" \
					--gpt_config "GPT_alef" \
					--save_prefix ${SP} \
					--epochs 5 \
					--batch_size 1 \
					--num_workers 8 \
					--optimizer "Adam" \
					--lr ${LR} \
					--seed ${SD} \
					--resume "/scratch/eo41/visual-recognition-memory/gpt_pretrained_models/imagenet_alef.pt"
			done	
		done	
	done	
done

echo "Done"
