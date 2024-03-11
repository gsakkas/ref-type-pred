#!/bin/bash

python -W ignore src/finetune/finetune.py\
  --model_path="bigcode/starcoderbase-3b"\
  --dataset_name="./benchmarks/lh_training_set_from_the_stack_without_lh_turorial.hf"\
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
  --output_dir="/tmp3/gsakkas/checkpoints_the_stack_no_lh_tutorial_20_epochs_2500_warmup_steps"\
  --model_dir="/tmp3/gsakkas/huggingface"\
