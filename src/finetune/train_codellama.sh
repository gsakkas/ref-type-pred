#!/bin/bash

python -W ignore src/finetune/finetune.py\
  --model_path="codellama/CodeLlama-7b-hf"\
  --dataset_name="./benchmarks/lh_training_set_from_the_stack.hf"\
  --size_valid_set 100\
  --seq_length 1024\
  --num_epochs 20\
  --save_freq 10\
  --log_freq 10\
  --eval_freq 10\
  --batch_size 1\
  --input_column_name="bad"\
  --output_column_name="ground_truth_type"\
  --gradient_accumulation_steps 8\
  --learning_rate 1e-4\
  --lr_scheduler_type="cosine"\
  --num_warmup_steps 2500\
  --weight_decay 0.05\
  --lora_r 16\
  --lora_alpha 32\
  --lora_dropout 0.05\
  --output_dir="/tmp3/gsakkas/codellama_7b_checkpoints_the_stack_20_epochs_2500_warmup_steps_rank_16_alpha_32_dropout_0_05"\
  --model_dir="/tmp3/gsakkas/huggingface"\
