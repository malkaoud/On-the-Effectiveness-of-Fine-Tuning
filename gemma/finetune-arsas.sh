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
    python3 -u finetune_arsas.py \
        --batch_size 1 \
        --lr 2e-4 \
        --epochs 3 \
        --seed $SEED \
        --test_file ArSAS-test.csv \
        --train_file training_data/arSAS_train_data_500v2.jsonl \
        --save_dir "arsas_saved_model_run_${SEED}" \
        --pred_csv "arsas_predictions_${SEED}.csv"
done