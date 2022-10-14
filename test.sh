#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --job-name=test_igpt
#SBATCH --output=test_igpt_%A_%a.out

module purge
module load cuda/11.3.1

# CONDITIONS=(novel_seen novel_unseen 1_exemplar_seen 1_exemplar_unseen 2_exemplar_seen 2_exemplar_unseen 4_exemplar_seen 4_exemplar_unseen 8_exemplar_seen 8_exemplar_unseen 16_exemplar_seen 16_exemplar_unseen)
# MODEL_DIRS=(saycam_1_0 saycam_1_1 imagenet_1_0 imagenet_1_1 tabularasa_1_0 tabularasa_1_1)
# MODEL_DIR=${MODEL_DIRS[$SLURM_ARRAY_TASK_ID / 12]}
# CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID % 12]}"

# echo $MODEL_DIR
# echo $CONDITION

python -u /scratch/eo41/visual-recognition-memory/test.py \
    --data_path /scratch/eo41/brady_1/test/state_unseen \
    --batch_size 100 \
    --model_dir /vast/eo41/visual-recognition-memory-models/saycam \
    --save_name losses_on_test_state_unseen_saycam

echo "Done"
