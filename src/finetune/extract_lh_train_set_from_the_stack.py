from os import listdir
from os.path import join
import re
from random import shuffle
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from difflib import SequenceMatcher


def clean_type(type):
    return ' '.join(type.split()).replace('{ ', '{').replace(' }', '}').strip()

# Get all ground truth liquid types from our Liquid Haskell tutorial
path_to_corrects_progs = "./benchmarks/correct_baselines"
lh_tutorial_types = set()
for target_file in listdir(path_to_corrects_progs):
    if not target_file.startswith("Ex") or "_correct" not in target_file:
        continue
    if not target_file.endswith(".hs") and not target_file.endswith(".lhs"):
        continue
    print(f"Getting {target_file} ground truth type...")
    with open(join(path_to_corrects_progs, target_file), "r", encoding="utf-8") as good_fin:
        fix_prog = good_fin.read()

    ground_truth = re.findall(r"{-@\s*?\S*?\s*?::[\s\S]*?@-}", fix_prog)
    for lh_type in ground_truth:
        # print(clean_type(lh_type))
        lh_tutorial_types.add(clean_type(lh_type))
print(len(lh_tutorial_types))

# Load the Haskell part of the Stack
ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/haskell", cache_dir="/tmp3/gsakkas/huggingface", split="train")

FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"

total = 0
count = 0
all_lh_types = 0
in_test_set = 0
unq_lh_types = set()
unq_lh_types_also_in_tutorial = set()
json_train_dataset = []
json_validation_dataset = []
print("Similar liquid types from LH tutorial:")
for dss in ds:
    if "{-@" in dss["content"]:
        ground_truth = re.findall(r"{-@\s*?\S*?\s*?::[\s\S]*?@-}", dss["content"])
        total += 1
        if ground_truth:
            count += 1
            all_lh_types += len(ground_truth)
            filename = dss['max_stars_repo_path']
            # if '\\' in filename:
            #     filename = filename.split('\\')[-1]
            starcoder_prompt = f"<filename>solutions/{filename}\n-- Fill in the masked refinement type in the following LiquidHaskell program\n"
            for lh_type in ground_truth:
                sample_pair = {}
                sim_ratio = max([SequenceMatcher(None, lh_type, gr_type).ratio() for gr_type in lh_tutorial_types])
                if lh_type not in unq_lh_types and sim_ratio < 0.7:
                    unq_lh_types.add(lh_type)
                    clean_lh_type = lh_type.split("::")[1].strip()
                    bad_prog = starcoder_prompt + dss["content"]
                    bad_prog = bad_prog.split(clean_lh_type)
                    sample_pair["bad"] = f"{FIM_PREFIX}{bad_prog[0]}{FIM_SUFFIX}{bad_prog[1]}{FIM_MIDDLE}"
                    sample_pair["ground_truth_type"] = f"{clean_lh_type}<|endoftext|>"
                    json_train_dataset.append(sample_pair)
                elif lh_type not in unq_lh_types_also_in_tutorial and not (sim_ratio < 0.7):
                    unq_lh_types_also_in_tutorial.add(lh_type)
                    print(lh_type)
                    clean_lh_type = lh_type.split("::")[1].strip()
                    bad_prog = starcoder_prompt + dss["content"]
                    bad_prog = bad_prog.split(clean_lh_type)
                    sample_pair["bad"] = f"{FIM_PREFIX}{bad_prog[0]}{FIM_SUFFIX}{bad_prog[1]}{FIM_MIDDLE}"
                    sample_pair["ground_truth_type"] = f"{clean_lh_type}<|endoftext|>"
                    json_validation_dataset.append(sample_pair)

print("\nTraining set statistics:")
print(f"{len(ds)}\tHaskell programs")
print(f"{total}\tHaskell programs with `{{-@` comments")
print(f"{count}\tLiquid Haskell programs")
print(f"{all_lh_types}\tLiquid Haskell types")
print(f"{len(unq_lh_types)}\tUnique Liquid Haskell types")
print(f"{len(unq_lh_types_also_in_tutorial)}\tLiquid Haskell types similar to LH tutorial")

# shuffle(json_train_dataset)
df_train_dataset = pd.DataFrame(json_train_dataset)
train_dataset = Dataset.from_pandas(df_train_dataset)
df_validation_dataset = pd.DataFrame(json_validation_dataset)
validation_dataset = Dataset.from_pandas(df_validation_dataset)

ds = DatasetDict()
ds["train"] = train_dataset
ds["validation"] = validation_dataset
ds.save_to_disk("./benchmarks/lh_training_set_from_the_stack_without_lh_turorial.hf")
