#!/bin/zsh

python -W ignore finetune.py\
  --model_path="bigcode/starcoderbase-3b"\
  --dataset_name="lh_train_set_from_the_stack.hf"\
  --size_valid_set 100\
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
  --num_warmup_steps 500\
  --weight_decay 0.05\
  --output_dir="/tmp3/gsakkas/checkpoints_the_stack_20_epochs"\
  --model_dir="/tmp3/gsakkas/huggingface"\
