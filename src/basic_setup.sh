git clone https://github.com/gsakkas/ref-type-pred.git
cd ref-type-pred
sudo apt-get -y install vim nvidia-cuda-toolkit
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade huggingface_hub
pip install -r src/requirements.txt
vim get_starcoder_code_suggestions.py
export OPENAI_KEY="dummy"
export PYTHONPATH=$PYTHONPATH:/home/gsakkas/Documents/UCSD/Program-Analytics/ref-type-pred/src
huggingface-cli login --token dummy # TODO: change dummy with actual token
CUDA_VISIBLE_DEVICES=0 python run_llm_generation.py --llm starcoderbase-3B --data_file lh_test_set_final_raw_v3.jsonl --total_repairs 20 --cache_file lh_starcoderbase_3B_finetuned_with_the_stack_chkpnt_20000_cache_raw_v3.json --create_cache_only