#!/bin/bash

# Number of runs (change as needed)
NUM_RUNS=10

# Set the master seed for reproducibility
MASTER_SEED=123

# Generate N random seeds with python and store in an array
SEEDS=($(python3 -c "import random; random.seed($MASTER_SEED); print(' '.join(str(random.randint(0, 2_147_483_647)) for _ in range($NUM_RUNS)))"))

for i in $(seq 0 $(($NUM_RUNS - 1)))
do
    SEED=${SEEDS[$i]}
    echo "Run $((i+1)): Using seed $SEED"
    python3 -u finetune_arsar.py \
        --batch_size 1 \
        --lr 2e-4 \
        --epochs 3 \
        --seed $SEED \
        --test_file ArSarcasm_test_.csv \
        --train_file training_data/ArSarcasm_train_data_500.jsonl \
        --save_dir "arsarcasm_saved_model_run_${SEED}" \
        --pred_csv "arsarcasm_predictions_${SEED}.csv"
done