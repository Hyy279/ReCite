#!/bin/bash
set -x

# Set up CUDA and memory management environment variables
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline 

# Define directories and file paths (anonymized)
BASE_DIR="/path/to/your_project_dir"
MODEL_PATH="/path/to/your_model_weights"
DATA_PATH="$BASE_DIR/train.parquet"
OUTPUT_DIR="$BASE_DIR/ReCite_qwen3_4b_sft"

mkdir -p $OUTPUT_DIR

# Launch distributed training via torchrun
torchrun --nproc_per_node=2 --master_port=29507 -m verl.trainer.sft_trainer \
    ++data.train_files=$DATA_PATH \
    ++data.val_files=$DATA_PATH \
    ++data.prompt_key="messages" \
    ++data.response_key="messages" \
    ++data.max_length=9216 \
    ++data.max_token_len_per_gpu=9216 \
    ++data.truncation="right" \
    ++data.train_batch_size=16 \
    ++model.path=$MODEL_PATH \
    ++model.enable_gradient_checkpointing=True \
    ++engine.model_dtype=bfloat16 \
    ++engine.dtype=bfloat16 \
    ++trainer.train_batch_size=16 \
    ++trainer.micro_batch_size=1 \
    ++trainer.total_epochs=3 \
    ++trainer.lr=2e-5 \
    ++optim.lr=2e-5 \
    ++optim.lr_warmup_steps_ratio=0.1 \
    ++optim.lr_scheduler_type="cosine" \
    ++trainer.default_local_dir=$OUTPUT_DIR \
    ++trainer.default_hdfs_dir=null \
    ++trainer.save_interval=125 \
    ++trainer.save_freq=125 \
    ++trainer.experiment_name=citeagent_sft \
    ++trainer.project_name=citeagent_research \
    ++trainer.logger=['console','tensorboard']
