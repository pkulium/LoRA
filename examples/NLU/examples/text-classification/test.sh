git pull
TASK_NAMES=("rte")
BATCH_SIZE=32  # This variable is now used in the script

# Define learning rates with more diversity
LEARNING_RATES=("2e-3" "4e-3" "6e-3" "8e-3" "2e-4" "4e-4" "6e-4" "8e-4" "2e-5" "4e-5" "6e-5" "8e-5")
LEARNING_RATES_MASKS=("1e-3" "3e-3" "5e-3" "7e-3" "1e-4" "3e-4" "5e-4" "7e-4" "1e-5" "3e-5" "5e-5" "7e-5")

for TASK_NAME in "${TASK_NAMES[@]}"
do
    MODEL="bert-base-cased"  # Assuming you want to use this model for all tasks
    for LEARNING_RATE in "${LEARNING_RATES[@]}"
    do
        for LEARNING_RATE_MASK in "${LEARNING_RATES_MASKS[@]}"
        do
            python -m pdb run_glue_no_trainer.py \
            --model_name_or_path $MODEL \
            --task_name $TASK_NAME \
            --gradient_accumulation_steps=1 \
            --per_device_train_batch_size $BATCH_SIZE \
            --per_device_eval_batch_size=128  \
            --learning_rate $LEARNING_RATE \
            --learning_rate_mask $LEARNING_RATE_MASK \  # Make sure this argument is correct
            --num_train_epochs 5 \
            --output_dir ./output/$TASK_NAME/$LEARNING_RATE/$LEARNING_RATE_MASK
        done
    done
done
