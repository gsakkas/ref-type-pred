import argparse
from os.path import exists, join
from os import listdir
import json
import re
import subprocess as subp
from predict.get_starcoder_code_suggestions import StarCoderModel
from collections import Counter

def get_args():
    parser = argparse.ArgumentParser(description='run_dependency_lh_tests')
    parser.add_argument('--total_preds', default=10, type=int,
                        help='total type predictions to generate per function type (default: 10)')
    parser.add_argument('--max_preds', default=50, type=int,
                        help='maximum number of predictions to generate with the model in total (default: 50)')
    parser.add_argument('--llm', default="starcoderbase-3b",
                        help='llm to use for code generation {incoder-1B, -6B, codegen25-7B, starcoderbase-1b, -3b, -15b} (default: starcoderbase-3b)')
    parser.add_argument('--update_cache', action='store_true',
                        help='update the prompt cache (default: False)')
    parser.add_argument('--use_cache', action='store_true',
                        help='use the prompt cache (default: False)')
    parser.add_argument('--create_cache_only', action='store_true',
                        help='only create the prompt cache and don\'t run any tests (default: False)')
    parser.add_argument('--cache_file', default="lh_dependency_tests_starcoderbase_3b_cache.json",
                        help='use the given file for prompt -> generation cache (default: simple starcoderbase-3b cache)')
    parser.add_argument('--print_logs', action="store_true", default=False,
                        help='print the log messages (default: False)')
    parser.add_argument('--exec_dir', default="../liquidhaskell/lh_exercises/dependency_tests",
                        help='benchmark data directory for execution (default: liquidhaskell/lh_exercises)')
    parser.add_argument('--out_dir', default="./results",
                        help='output data directory (default: ./results)')
    _args = parser.parse_args()
    return _args


def make_prompt_from_masked_code(badp, mask_id, masked_types, filename):
    FIM_PREFIX = "<fim_prefix>"
    FIM_MIDDLE = "<fim_middle>"
    FIM_SUFFIX = "<fim_suffix>"

    prefix = f"<filename>solutions/{filename}\n-- Fill in the masked refinement type in the following LiquidHaskell program\n"
    # Remove all other masks and keep the one we want to predict for
    one_mask_bad_prog = badp.replace(f"<mask_{mask_id}>", "<fimask>")
    for mtype in masked_types:
        if masked_types[mtype] in one_mask_bad_prog:
            one_mask_bad_prog = re.sub(masked_types[mtype] + r"\s*", "", one_mask_bad_prog, 1)
    split_code = one_mask_bad_prog.split("<fimask>")
    prompt = f"{FIM_PREFIX}{prefix}{split_code[0]}{FIM_SUFFIX}{split_code[1]}{FIM_MIDDLE}"

    return prompt


def get_type_predictions(prompt, key, llm, args):
    print("-" * 42)
    print(f"New type predictions for {key}")
    print("-> LLM generation...", flush=True)
    # NOTE: The deeper we are in the loop, get more predictions
    # in order to avoid backtracking and potentially removing a correct earlier type
    # Potentially, can be done differently, i.e. generate more when failing
    prog_preds = llm.get_code_suggestions(prompt, args.max_preds)
    freq_map = Counter(prog_preds)
    freq_sum = sum(freq for _, freq in freq_map.most_common(args.total_preds))
    prog_preds = [(pred, freq * 100 // freq_sum) for pred, freq in freq_map.most_common(args.total_preds)]
    print(prog_preds)
    total_prob = sum(prob for _, prob in prog_preds)
    print(total_prob)
    if total_prob < 100:
        preds, prob = prog_preds[0]
        prog_preds[0] = (preds, prob + 100 - total_prob)
    total_prob = sum(prob for _, prob in prog_preds)
    print(total_prob)
    print(f"-> {len(prog_preds)} unique predicted types", flush=True)
    return prog_preds


def replace_type_with_pred(func, pred, prog):
    # NOTE: need to take care of possible '\f ->' in predicted types
    # '\\'s are evaluated to one '\' when subing, so we need to add another one
    if "\\" in pred:
        pred = pred.replace("\\", "\\\\")
    llm_prog = re.sub(r"{-@\s*?" + func + r"\s*?::[\s\S]*?@-}", pred, prog, 1)
    return llm_prog


def restore_mask_at_id(m_id, func, prog):
    print(f"Restored {func} :: <mask_{m_id}>")
    llm_prog = re.sub(r"{-@\s*?" + func + r"\s*?::[\s\S]*?@-}", f"{{-@ {func} :: <mask_{m_id}> @-}}", prog, 1)
    return llm_prog


def restore_ignored_masks(all_mtypes, prog):
    llm_prog = prog
    for mtype in all_mtypes:
        ignore_func = all_mtypes[mtype].split()[1].strip()
        if f"{{-@ ignore {ignore_func} @-}}" in llm_prog:
            # print(f"Restored ignore {all_mtypes[mtype]}...")
            llm_prog = llm_prog.replace(f"{{-@ ignore {ignore_func} @-}}", all_mtypes[mtype], 1)
    return llm_prog


def lh_verifies_prog(prog, target_file, args):
    with open(join(args.exec_dir, target_file.replace(".hs", "_llm.hs")), "w", encoding="utf-8") as llm_fin:
        llm_fin.write(prog)

    solved = False
    cmds = "source /home/gsakkas/.ghcup/env; " # for local Haskell installation
    cmds += "export PATH=$PATH:/home/gsakkas/usr/bin; " # for local Z3 installation
    cmds += f"cd {args.exec_dir}; "
    cmds += f"stack exec ghc -- -fplugin=LiquidHaskell {target_file.replace('.hs', '_llm.hs')}"
    test_output = subp.run(cmds, shell=True, check=False, capture_output=True)
    result = test_output.stdout.decode('utf-8').strip()
    if result != "" and "UNSAFE" not in result and "SAFE" in result:
        solved = True
    return solved


def run_tests(path, args):
    # diffsToStr = {0: "Easy", 1: "Medium", 2: "Hard"}
    difficulties = {
        "Ex9.hs": 0,
        "Ex10.hs": 1,
        "Ex11.hs": 2,
        "Ex12.hs": 2
    }
    fixed_progs = 0
    all_progs = 0
    masks_per_exer = {k: 0 for k in set(difficulties.keys())}
    correct_llm_types_per_exer = {k: 0 for k in set(difficulties.keys())}
    deepest_correct_type_id = {k: 0 for k in set(difficulties.keys())}

    # Load (finetuned) code LLM
    code_llm = StarCoderModel()

    for target_file in sorted(listdir(path))[3:]:
        if not target_file.startswith("Ex") or "_correct" in target_file:
            continue
        if not target_file.endswith(".hs") and not target_file.endswith(".lhs"):
            continue
        print("=" * 42 * 2)
        print(target_file)
        all_progs += 1
        with open(join(path, target_file), "r", encoding="utf-8") as bad_fin:
            bad_prog = bad_fin.read()
        with open(join(path, target_file.replace(".hs", "_correct.hs")), "r", encoding="utf-8") as good_fin:
            fix_prog = good_fin.read()

        all_liquid_types = re.findall(r"{-@[\s\S]*?@-}", bad_prog)
        masked_funcs = list(filter(lambda x: "<mask_" in x, all_liquid_types))

        ground_truths = {}
        masked_func_types = {}
        for masked_type in masked_funcs:
            func = masked_type.split()[1].strip()
            ground_truth = re.findall(r"{-@\s*?" + func + r"\s*?::([\s\S]*?)@-}", fix_prog)[0].strip()
            ground_truths[func] = ground_truth
            mask_id = int(masked_type.split("<mask_")[1].split(">")[0].strip())
            masked_func_types[mask_id] = masked_type

        masks_per_exer[target_file] = len(masked_func_types)
        type_preds_cache = {}
        llm_prog = bad_prog
        mask_id = 1
        current_type_state = ["-- No refinement type"] * (len(masked_func_types) + 1) # + 1 so we don't care about 0 element
        current_type_state[0] = ""
        seen_states = {}
        correct_llm_types_per_exer[target_file] = len(masked_func_types)
        error_counter = 0
        while mask_id > 0:
            masked_type = masked_func_types[mask_id]
            func = masked_type.split()[1].strip()
            print("=" * 42)
            print(f"Solving {target_file} ({func})...", flush=True)
            key = f"{target_file}--{func}"
            # NOTE: Exit criteria for the backtracking loop
            # If we have generated too many types for this function
            if mask_id in type_preds_cache and type_preds_cache[mask_id][1] >= 100 and \
                    all(type_preds_cache[k][1] >= 100 for k in type_preds_cache):
                print(f"Reached limit of predictions; Exiting...", flush=True)
                break
            if mask_id not in type_preds_cache or (type_preds_cache[mask_id][0] == [] and type_preds_cache[mask_id][1] < 100):
                if mask_id in type_preds_cache:
                    print(type_preds_cache[mask_id])
                llm_prog = restore_mask_at_id(mask_id, func, llm_prog)
                prompt = make_prompt_from_masked_code(llm_prog, mask_id, masked_func_types, target_file)
                mtype_preds = get_type_predictions(prompt, key, code_llm, args)
                # if mask_id in type_preds_cache:
                #     type_preds_cache[mask_id] = mtype_preds, type_preds_cache[mask_id][1] + args.total_preds
                # else:
                type_preds_cache[mask_id] = mtype_preds, 0

            testing_new_type = False
            while type_preds_cache[mask_id][0]:
                type_prediction, prob = type_preds_cache[mask_id][0].pop(0)
                preds, total_prob = type_preds_cache[mask_id]
                type_preds_cache[mask_id] = (preds, total_prob + prob)
                # NOTE: Just a random check, cause LH crashes for too long types
                if len(type_prediction) > len(ground_truths[func]) + 32:
                    print("Type too long!")
                    continue
                current_type_state[mask_id] = f"{{-@ {func} :: {type_prediction} @-}}"
                if "".join(current_type_state) in seen_states:
                    print("Skipping....")
                    continue
                testing_new_type = True
                break
            if not testing_new_type and type_preds_cache[mask_id][1] < 100:
                error_counter += 1
                if error_counter > masks_per_exer[target_file] * 100:
                    # Probably got stuck on a infinite loop
                    print("Something went very wrong... Exiting...")
                    break
                continue
            elif type_preds_cache[mask_id][1] >= 100 and current_type_state[mask_id] != ground_truths[func]:
                current_type_state[mask_id] = f"{{-@ {func} :: {ground_truths[func]} @-}}"
                if "".join(current_type_state) in seen_states:
                    min_prob = 100
                    min_mask = -1
                    for mid in type_preds_cache:
                        if type_preds_cache[mid][1] < min_prob:
                            min_prob = type_preds_cache[mid][1]
                            min_mask = mid
                    mask_id = min_mask
                    print(f"Backtracking to mask id = {mask_id}...", flush=True)
                    continue
                print(f"Testing the ground truth type, since we reached max limit of predictions...", flush=True)
                correct_llm_types_per_exer[target_file] -= 1
            all_types_replaced = all([type_pred != "-- No refinement type" for type_pred in current_type_state])
            if all_types_replaced and "".join(current_type_state) not in seen_states:
                print("-" * 42)
                for type_pred in current_type_state[1:]:
                    t_func = type_pred.split()[1].strip()
                    print(f"Testing {type_pred}...", flush=True)
                    llm_prog = replace_type_with_pred(t_func, type_pred, llm_prog)

                solved = False
                if lh_verifies_prog(llm_prog, target_file, args):
                    solved = True
                    break
                seen_states["".join(current_type_state)] = solved
                print("-" * 42)

                if solved:
                    print("...SAFE", flush=True)
                    break
                else:
                    print("...UNSAFE", flush=True)
            if all_types_replaced:
                # Always jumpt to mask id with min probability
                min_prob = 100
                min_mask = -1
                for mid in type_preds_cache:
                    if type_preds_cache[mid][1] < min_prob:
                        min_prob = type_preds_cache[mid][1]
                        min_mask = mid
                mask_id = min_mask
                print(f"Jumping to mask id = {mask_id}...", flush=True)
            else:
                mask_id += 1

        if mask_id > 0 and correct_llm_types_per_exer[target_file] == masks_per_exer[target_file]:
            fixed_progs += 1
        else:
            correct_llm_types_per_exer[target_file] = 0
        print("=" * 42)
        print(f"{correct_llm_types_per_exer[target_file]} / {masks_per_exer[target_file]} types predicted correctly")

    print("=" * 42)
    print("=" * 42)
    for k in sorted(correct_llm_types_per_exer.keys()):
        print(f">>> Chapter {k[2:].split('.')[0]}")
        print(f"{correct_llm_types_per_exer[k]} / {masks_per_exer[k]} types predicted correctly")
        print(f"{correct_llm_types_per_exer[k] * 100 / masks_per_exer[k]:.2f}% prediction accuracy")
        print("-" * 42)
    print("=" * 42)
    print("=" * 42)
    print(f"{fixed_progs} / {all_progs} programs fully annotated correctly with LH types")


if __name__ == "__main__":
    cmd_args = get_args()

    run_tests("benchmarks/dependency_tests", cmd_args)