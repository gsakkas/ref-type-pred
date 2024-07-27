git clone https://github.com/gsakkas/ref-type-pred.git
git clone git@github.com:Z3Prover/z3.git
# Install z3 per their instructions
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
git clone git@github.com:oxarbitrage/hsalsa20.git
cd hsalsa20
git checkout 145bdfa21f078097695f1fe23565e58eb6d54a0d
cd ..
git clone git@github.com:pratyush401/bytestring_lh-llm.git
cd bytestring_lh-llm
git checkout final_version
cd ..
# sudo apt-get -y install vim nvidia-cuda-toolkit
cd ref-type-pred
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r src/requirements.txt
cd src
git clone git@github.com:tree-sitter/tree-sitter-haskell.git
cd ..
source /home/gsakkas/.ghcup/env # for local Haskell installation
export PATH=$PATH:/home/gsakkas/usr/bin # for local Z3 installation
export OPENAI_KEY="dummy"
export PYTHONPATH=$PYTHONPATH:/home/gsakkas/ref-type-pred/src
huggingface-cli login --token dummy # TODO: change dummy with actual token
CUDA_VISIBLE_DEVICES=0 python src/predict/run_llm_generation.py --llm starcoderbase-3B --use_finetuned --data_file benchmarks/lh_test_set_final_with_haskell_types_and_tests_v3.jsonl --total_repairs 50 --cache_file benchmarks/lh_finetuned_v2_starcoder_3b_cache_with_haskell_types_and_tests_v3.json --create_cache_only
CUDA_VISIBLE_DEVICES=0 python src/evaluation/run_lh_dependency_test_set_with_repairs.py --total_preds 10 --max_preds 50 > results/lh_dependency_tests_starcoderbase_3b_test.txt