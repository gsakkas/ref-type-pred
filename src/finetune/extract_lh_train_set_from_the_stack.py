import re
from random import shuffle
import pandas as pd
from datasets import load_dataset, Dataset

ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/haskell", split="train")

FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"

total = 0
count = 0
all_lh_types = 0
unq_lh_types = set()
json_dataset = []
for dss in ds:
    if "{-@" in dss["content"]:
        # if total > 5:
        #     break
        # print("=" * 42)
        # print(dss["content"])
        ground_truth = re.findall(r"{-@\s*?(\S*?\s*?::[\s\S]*?)@-}", dss["content"])
        # print("-" * 42)
        # print(ground_truth)
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
                if lh_type not in unq_lh_types:
                    unq_lh_types.add(lh_type)
                    clean_lh_type = lh_type.split("::")[1].strip()
                    bad_prog = starcoder_prompt + dss["content"]
                    bad_prog = bad_prog.split(clean_lh_type)
                    sample_pair["bad"] = f"{FIM_PREFIX}{bad_prog[0]}{FIM_SUFFIX}{bad_prog[1]}{FIM_MIDDLE}"
                    sample_pair["ground_truth_type"] = f"{clean_lh_type}<|endoftext|>"
                    json_dataset.append(sample_pair)

print(f"{len(ds)}\tHaskell programs")
print(f"{total}\tHaskell programs with `{{-@` comments")
print(f"{count}\tLiquid Haskell programs")
print(f"{all_lh_types}\tLiquid Haskell types")
print(f"{len(unq_lh_types)}\tUnique Liquid Haskell types")

# shuffle(json_dataset)
df_dataset = pd.DataFrame(json_dataset)
print(df_dataset)
train_dataset = Dataset.from_pandas(df_dataset.rename(columns={0: "train"}), split="train")
train_dataset.save_to_disk("lh_train_set_from_the_stack.hf")
print(train_dataset[0])
