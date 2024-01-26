#!/bin/zsh

torchrun --nproc_per_node 3 finetune.py \
  --model_path="bigcode/starcoderbase-3b"\
  --dataset_name="lh_train_set_final_with_haskell_types_v3.hf"\
  --size_valid_set 100\
  --seq_length 2048\
  --max_steps 500\
  --save_freq 50\
  --log_freq 50\
  --eval_freq 50\
  --batch_size 1\
  --input_column_name="bad"\
  --output_column_name="ground_truth_type"\
  --gradient_accumulation_steps 8\
  --learning_rate 1e-4\
  --lr_scheduler_type="cosine"\
  --num_warmup_steps 100\
  --weight_decay 0.05\
  --output_dir="/tmp3/gsakkas/checkpoints_parallel"\
  --model_dir="/tmp3/gsakkas/huggingface"\
