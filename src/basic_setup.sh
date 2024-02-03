git clone https://github.com/gsakkas/ref-type-pred.git
cd ref-type-pred
sudo apt-get -y install vim nvidia-cuda-toolkit
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade huggingface_hub
pip install -r src/requirements.txt
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
source /home/gsakkas/.ghcup/env # for local Haskell installation
export PATH=$PATH:/home/gsakkas/usr/bin # for local Z3 installation
export OPENAI_KEY="dummy"
export PYTHONPATH=$PYTHONPATH:/home/gsakkas/Documents/UCSD/Program-Analytics/ref-type-pred/src
huggingface-cli login --token dummy # TODO: change dummy with actual token
# CUDA_VISIBLE_DEVICES=0 python run_llm_generation.py --llm starcoderbase-3B --data_file lh_test_set_final_raw_v3.jsonl --total_repairs 20 --cache_file lh_starcoderbase_3B_finetuned_with_the_stack_chkpnt_20000_cache_raw_v3.json --create_cache_only
CUDA_VISIBLE_DEVICES=0 python src/evaluation/run_lh_dependency_test_set_with_repairs.py --total_preds 10 --max_preds 50 > results/lh_dependency_tests_starcoderbase_3b_test.txt