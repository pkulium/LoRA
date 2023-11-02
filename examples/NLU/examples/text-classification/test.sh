TASK_NAMES=("rte")
BATCH_SIZE="32"
LEARNING_RATES=("5e-5" "7e-5" "1e-4")
ADMM_RHOS=("4e-3")

for TASK_NAME in ${TASK_NAMES[*]}
    do
    # MODEL=$HOME/$TASK_NAME/
    MODEL="bert-base-cased"

    for LEARNING_RATE in ${LEARNING_RATES[*]}
    do
        for ADMM_RHO in ${ADMM_RHOS[*]}
        do
            python run_glue.py \
            --model_name_or_path bert-base-cased \
            --task_name $TASK_NAME \
            --do_train \
            --do_eval \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --learning_rate $LEARNING_RATE \
            --num_train_epochs 10 \
            --output_dir /tmp/$TASK_NAME/
        done
    done
done