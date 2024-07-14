#!/bin/bash

python -W ignore src/finetune/finetune.py\
  --model_path="bigcode/starcoderbase-3b"\
  --dataset_name="./benchmarks/lh_training_set_from_the_stack_v2.hf"\
  --size_valid_set 1000\
  --seq_length 2048\
  --num_epochs 20\
  --save_freq 100\
  --log_freq 100\
  --eval_freq 100\
  --batch_size 1\
  --input_column_name="bad"\
  --output_column_name="ground_truth_type"\
  --gradient_accumulation_steps 8\
  --no_gradient_checkpointing\
  --learning_rate 1e-4\
  --lr_scheduler_type="cosine"\
  --num_warmup_steps 2500\
  --weight_decay 0.05\
  --lora_r 64\
  --lora_alpha 32\
  --lora_dropout 0.1\
  --output_dir="/tmp3/gsakkas/starcoder_3b_checkpoints_the_stack_v2_20_epochs_2500_warmup_steps_rank_64_alpha_32_dropout_0_1"\
  --model_dir="/tmp3/gsakkas/huggingface"\
