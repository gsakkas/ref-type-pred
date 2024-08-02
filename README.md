# Neurosymbolic Modular Refinement Type Inference

## Installation

First we need to install Haskell and it's ecosystem. The easiest way to do this
currently is:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
```
Check the official website of [GHCup](https://www.haskell.org/ghcup/#) for more
details and how to install Haskell for different OSes.

We showcase here how to run the evaluation only for the **HSalsa20** repository,
which is public and is *not* owned by us. The **ByteString** package, while it's
a public Haskell module, our *own forked* version is private for now. Since,
it's hard to anonymize that version, we omit it for the double-blinded review
process and we focus only on **HSalsa20** for now. We will make all our code,
fine-tuned models and data public after a paper acceptance.

Next, we need to clone the **HSalsa20** repository and checkout the commit we
performed our evaluation on:
```bash
git clone git@github.com:oxarbitrage/hsalsa20.git
cd hsalsa20
git checkout 145bdfa21f078097695f1fe23565e58eb6d54a0d
cd ..
```

*Liquid Haskell* relies on SMT solvers for program verification. We used *Z3*
and it can be found on github [here](https://github.com/Z3Prover/z3):
```bash
git clone git@github.com:Z3Prover/z3.git
```

The extended instructions to install *Z3* can be found there, but this is the
main way to do it, that worked for us across different machines:
```bash
python scripts/mk_make.py
cd build
make
sudo make install
```

Finally, let's install all *Python* dependencies for our tool *XO* (the code
repo is named `ref-type-pred` for now):
```bash
cd ref-type-pred
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r src/requirements.txt
cd src
git clone git@github.com:tree-sitter/tree-sitter-haskell.git
cd ..
export PYTHONPATH=$PYTHONPATH:/path/to/ref-type-pred/src
```

Depending on the environment that this code is executed on, the following
commands might be necessary:
```bash
source /path/to/.ghcup/env                    # for local Haskell installation
export PATH=$PATH:/path/to/usr/bin            # for local Z3 installation
huggingface-cli login --token "YOUR_HF_TOKEN" # TODO: change YOUR_HF_TOKEN with actual token
```

You will need a [huggingface](https://huggingface.co/) account in order to
create a *token* and access the LLMs that we used.

All the tests for **HSalsa20** are in the script `run_tests_hsalsa.sh` that can be directly run as:
```bash
bash run_tests_hsalsa.sh
```

This script will run all tests for all the combinations mentioned in the
*Evaluation* section of our paper using the **StarCoder 3B** LLM, which is the
smallest that we tested, thus it has the lowest memory requirement. You will
need however a GPU with *at least* 12 GBs of GPU VRAM, while **StarCoder 15B**
would require at least 40 GBs of GPU VRAM.

We are also not able to provide the **very large** fine-tuned weight files for
these LLMs at this preliminary artifact, without making them public on
hugginface or uploading enormous files with the submission, which would
jeopardized the anonymity of this submission. However, we will provide them and
make them public on a later stage, for the final artifact evaluation.

For the time being, the different optimizations discussed in the paper can be
ran with the baseline pre-trained LLMs, and while the results might not reach
the highs of our evaluation using the fine-tuned models, we expect to see
improvements over the baseline even with this configuration..