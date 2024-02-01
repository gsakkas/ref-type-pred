from os.path import join
from os import listdir
import re
from random import shuffle
import pandas as pd
from datasets import Dataset


KEEP_AND_COMMENT_NON_CODE = False
USE_BUGGY_LINES_AS_COMMENT = False
MASK_ONLY_POST_COND = False
USE_RAW_PROGRAMS = False # i.e. no haskell types and no tests
USE_HASKEL_TYPES = True # i.e. haskell types in programs, but no tests
USE_HASKEL_TYPES_AND_TESTS = False # i.e. haskell types and tests in programs
FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"

def comment_out_non_code(prog):
    """
    Comments out all non-code lines from the LiquidHaskell student exercises.
    Keeps code within \\begin{code} ... \\end{code} as it was.
    """
    prog_lines = prog.splitlines()
    upd_prog_lines = []
    comment = True
    for pline in prog_lines:
        updated_line = pline.rstrip()
        if "\\begin{code}" in pline:
            comment = False
            updated_line = "-- " + updated_line
        elif "\\end{code}" in pline:
            comment = True
        if comment:
            updated_line = "-- " + updated_line
        if not("<div" in pline or "</div" in pline):
            upd_prog_lines.append(updated_line)

    return "\n".join(upd_prog_lines)


def remove_non_code(prog):
    """
    Deletes all non-code lines from the LiquidHaskell student exercises.
    Keeps code within \\begin{code} ... \\end{code} as it was.
    """
    prog_lines = prog.splitlines()
    upd_prog_lines = []
    to_keep = False
    for pline in prog_lines:
        if "\\end{code}" in pline:
            to_keep = False
            upd_prog_lines.append("\n")
        if to_keep:
            upd_prog_lines.append(pline.rstrip())
        if "\\begin{code}" in pline:
            to_keep = True

    return "\n".join(upd_prog_lines)


# The prompt that the StarCoder paper claims to improve the results
starcoder_prompt = "<filename>solutions/{file_name}\n-- Fill in the masked refinement type in the following LiquidHaskell program\n"
path_to_corrects_progs = "../liquid_haskell_exercises/test"
path_to_bad_progs = path_to_corrects_progs
if USE_RAW_PROGRAMS:
    path_to_bad_progs += "/no_haskell_types_no_tests"
if USE_HASKEL_TYPES:
    path_to_bad_progs += "/no_tests"
if USE_HASKEL_TYPES_AND_TESTS:
    path_to_bad_progs += "/with_tests"
json_dataset = []
valid_pairs = 0
total_type_errors = 0
ground_truth_types = {}
samples_per_func = {}
for target_file in listdir(path_to_bad_progs):
    if not target_file.startswith("Ex") or "_correct" in target_file:
        continue
    if not target_file.endswith(".hs") and not target_file.endswith(".lhs"):
        continue
    print(target_file)
    with open(join(path_to_bad_progs, target_file), "r", encoding="utf-8") as bad_fin:
        bad_prog = bad_fin.read()
    with open(join(path_to_corrects_progs, target_file.replace(".hs", "_correct.hs")), "r", encoding="utf-8") as good_fin:
        fix_prog = good_fin.read()

    all_liquid_types = re.findall(r"{-@[\s\S]*?@-}", bad_prog)
    func = list(filter(lambda x: "<mask>" in x, all_liquid_types))[0].strip().split()[1].strip()
    if func in samples_per_func and samples_per_func[func] > 9:
        continue
    print("++++++++++++++++++++++++++++++")
    print(f">>> {func}")

    # if KEEP_AND_COMMENT_NON_CODE:
    #     # new_bad_prog = comment_out_non_code(bad_prog).replace(" LH ", " LiquidHaskell ")
    #     new_bad_prog = comment_out_non_code(bad_prog).replace(" LH ", " LiquidHaskell ")
    # else:
    #     # new_bad_prog = remove_non_code(bad_prog).replace(" LH ", " LiquidHaskell ")
    #     new_bad_prog = remove_non_code(bad_prog).replace(" LH ", " LiquidHaskell ")
    # # Simple clean-up
    # # new_bad_prog = new_bad_prog.replace(" LH ", " LiquidHaskell ").replace("\n\n\n", "\n\n")
    # new_bad_prog = new_bad_prog.replace(" LH ", " LiquidHaskell ").replace("\n\n\n", "\n\n")

    ground_truth = re.findall(r"{-@\s*?" + func + r"\s*?::([\s\S]*?)@-}", fix_prog)
    if not ground_truth:
        continue
    new_bad_prog = re.sub(r"{-@\s*?" + func + r"\s*?::([\s\S]*?)@-}", "{-@ " + func + " :: <fimask> @-}", bad_prog)

    if MASK_ONLY_POST_COND:
        return_type = ground_truth[0].split("->")[-1].strip()
        pre_cond = ' -> '.join([typ.strip() for typ in ground_truth[0].split("->")[:-1]])
        new_bad_prog = new_bad_prog.replace("<fimask>", pre_cond + " -> <fimask>")

    sample_pair = {}
    valid_pairs += 1
    prefix = starcoder_prompt.format(file_name=target_file)
    sample_pair["key"] = f"{target_file}--{func}"
    sample_pair["prefix"] = prefix
    sample_pair["masked_code"] = prefix + new_bad_prog
    if sample_pair["masked_code"].count("<fimask>") != 1:
        print(f"{sample_pair['masked_code'].count('<fimask>')} masks in final prompt")
    split_code = sample_pair["masked_code"].split("<fimask>")
    sample_pair["bad"] = f"{FIM_PREFIX}{split_code[0]}{FIM_SUFFIX}{split_code[1]}{FIM_MIDDLE}"
    sample_pair["fix"] = new_bad_prog
    # print(sample_pair["bad"])
    total_type_errors += 1
    sample_pair["ground_truth_type"] = ground_truth[0].strip() if ground_truth else "deleted_in_fixed_prog"
    ground_truth_types[sample_pair["key"]] = sample_pair["ground_truth_type"].splitlines()
    if func in samples_per_func:
        samples_per_func[func] += 1
    else:
        samples_per_func[func] = 1
    sample_pair["ground_truth_type"] += "<|endoftext|>"
    json_dataset.append(sample_pair)

print(f"{valid_pairs} valid pairs")
print(f"{total_type_errors} / {valid_pairs} type error pairs")
print(samples_per_func)

file_suffix = ""
if USE_RAW_PROGRAMS:
    file_suffix += "_raw_v3"
if USE_HASKEL_TYPES:
    file_suffix += "_with_haskell_types_v3"
if USE_HASKEL_TYPES_AND_TESTS:
    file_suffix += "_with_haskell_types_and_tests_v3"


# Just to make a "fake" bigger dataset
json_dataset = json_dataset * 10
# shuffle(json_dataset)
df_dataset = pd.DataFrame(json_dataset)
train_dataset = Dataset.from_pandas(df_dataset.rename(columns={0: "train"}), split="train")
train_dataset.save_to_disk(f"lh_train_set_final{file_suffix}.hf")
print(train_dataset[0])
