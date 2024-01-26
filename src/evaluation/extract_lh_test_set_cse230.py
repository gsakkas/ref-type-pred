from os.path import join, basename
import ast
import json
import re

KEEP_AND_COMMENT_NON_CODE = False
USE_BUGGY_LINES_AS_COMMENT = False


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

all_funcs = ["foldr1", "kmeans1", "replicate", "mergeCluster", "collapse",
            "concat", "nearest", "expand", "length", "zipWith", "kmeans",
            "centroid", "mapReduce", "add", "concat2", "distance", "map", "group"]

# The prompt that the StarCoder paper claims to improve the results
starcoder_prompt = "<filename>solutions/{file_name}\n-- Fill the refinement type errors in the following LiquidHaskell code exercise\n"
path_to_dataset = "../../pldi2019-counterfactual-symex"
json_dataset = []
ground_truth_types = {}
valid_pairs = 0
total_type_errors = 0
seen_set = set()
samples_per_func = {k: 0 for k in all_funcs}
with open(join(path_to_dataset, "G2/benchmarks-env/eval-tests/dump-good.txt"), "r", encoding="utf-8") as good_fin, \
    open(join(path_to_dataset, "G2/benchmarks-env/eval-tests/dump-etc.txt"), "r", encoding="utf-8") as etc_fin:
    txt_pairs = good_fin.read() + etc_fin.read()
    txt_lines = txt_pairs.split("ID:")[1:]
    for chunk in txt_lines:
        ch_lines = chunk.splitlines()[1:-2]
        bad_dir = join("G2", ch_lines[0].split("G2")[-1][1:])
        fix_dir = join("G2", ch_lines[1].split("G2")[-1][1:])
        failed_prop, bad_func = ast.literal_eval(ch_lines[2])
        with open(join(path_to_dataset, bad_dir), "r", encoding="utf-8") as bad_fin:
            bad_prog = bad_fin.read()
        with open(join(path_to_dataset, fix_dir), "r", encoding="utf-8") as good_fin:
            fix_prog = good_fin.read()

        all_liquid_types = re.findall(r"{-@[\s\S]*?@-}", bad_prog)
        for func in all_funcs:
            if samples_per_func[func] > 9:
                continue
            if all(func not in lht for lht in all_liquid_types):
                # print(f"++>>> {func}")
                continue
            # print(f"-->>> {func}")
            if KEEP_AND_COMMENT_NON_CODE:
                new_bad_prog = comment_out_non_code(bad_prog).replace(" LH ", " LiquidHaskell ")
                new_fix_prog = comment_out_non_code(fix_prog).replace(" LH ", " LiquidHaskell ")
            else:
                new_bad_prog = remove_non_code(bad_prog).replace(" LH ", " LiquidHaskell ")
                new_fix_prog = remove_non_code(fix_prog).replace(" LH ", " LiquidHaskell ")
            # Simple clean-up
            new_bad_prog = new_bad_prog.replace(" LH ", " LiquidHaskell ").replace("\n\n\n", "\n\n")
            new_fix_prog = new_fix_prog.replace(" LH ", " LiquidHaskell ").replace("\n\n\n", "\n\n")
            # print(sample_pair["bad"])

            new_bad_prog = re.sub(r"{-@\s*?" + func + r"\s*?::([\s\S]*?)@-}", "{-@ " + func + " :: <fimask> @-}", new_bad_prog)
            ground_truth = re.findall(r"{-@\s*?" + func + r"\s*?::([\s\S]*?)@-}", new_fix_prog)
            # print(ground_truth[0].strip())
            sample_pair = {}

            valid_pairs += 1
            fname = basename(bad_dir).replace("flycheck_", "").split(".lhs")[0] + ".lhs"
            prefix = starcoder_prompt.format(file_name=fname)
            # if USE_BUGGY_LINES_AS_COMMENT:
            #     prefix_mask = "\n".join([f"-- {ml}" for ml in m.splitlines()])
            #     new_mask = f"{prefix_mask}\n{new_mask}"
            if f"{new_bad_prog}\n{new_fix_prog}\n{func}" in seen_set:
                print("OUT!")
                continue
            seen_set.add(f"{new_bad_prog}\n{new_fix_prog}\n{func}")

            sample_pair["key"] = f"{bad_dir}--{fix_dir}--{func}"
            sample_pair["prefix"] = prefix
            sample_pair["masked_code"] = prefix + new_bad_prog
            if sample_pair["masked_code"].count("<fimask>") != 1:
                print(f"{sample_pair['masked_code'].count('<fimask>')} masks in final prompt")
                continue
            FIM_PREFIX = "<fim_prefix>"
            FIM_MIDDLE = "<fim_middle>"
            FIM_SUFFIX = "<fim_suffix>"
            split_code = sample_pair["masked_code"].split("<fimask>")
            sample_pair["bad"] = f"{FIM_PREFIX}{split_code[0]}{FIM_SUFFIX}{split_code[1]}{FIM_MIDDLE}"
            sample_pair["fix"] = new_fix_prog
            # print(sample_pair["bad"])
            print("++++++++++++++++++++++++++++++")
            total_type_errors += 1
            sample_pair["ground_truth_type"] = ground_truth[0].strip() if ground_truth else "deleted_in_fixed_prog"
            json_dataset.append(sample_pair)
            ground_truth_types[sample_pair["key"]] = sample_pair["ground_truth_type"].splitlines()
            print(f">>> {func}")
            samples_per_func[func] += 1
            if len(json_dataset) == 200:
                break
        if len(json_dataset) == 200:
            break

print(f"{valid_pairs} valid pairs")
print(f"{total_type_errors} / {valid_pairs} type error pairs")
print(f"{len(seen_set)} / {len(json_dataset)} unique type error pairs")
print(samples_per_func)

with open("liquid_haskell_test_set_cse_230.jsonl", "w", encoding="utf-8") as out_file:
    for sample_dict in json_dataset:
        out_file.write(json.dumps(sample_dict) + '\n')

with open("liquid_haskell_test_set_cse_230_labels.jsonl", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(ground_truth_types, indent=4) + '\n')
