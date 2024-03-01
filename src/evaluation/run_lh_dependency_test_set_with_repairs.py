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


def clean_type(type):
    return ' '.join(type.split()).replace('{ ', '{').replace(' }', '}').strip()


def get_type_predictions(prompt, key, func_name, ground_truth, llm, args):
    print("-" * 42)
    print(f"New type predictions for {key}")
    print("-> LLM generation...", flush=True)
    # NOTE: The deeper we are in the loop, get more predictions
    # in order to avoid backtracking and potentially removing a correct earlier type
    # Potentially, can be done differently, i.e. generate more when failing
    prog_preds = llm.get_code_suggestions(prompt, min(args.max_preds, args.total_preds))
    prog_preds = [clean_type(pred) for pred in prog_preds]
    prog_preds = list(filter(lambda p: p.count('->') == ground_truth.count('->'), prog_preds))
    freq_map = Counter(filter(lambda p: func_name not in p, prog_preds))
    prog_preds = [pred for pred, _ in freq_map.most_common(args.total_preds)]
    print(f"-> {len(prog_preds[:10])} unique predicted types", flush=True)
    return prog_preds[:10]


def replace_type_with_pred(func, pred, prog, all_mtypes):
    tp = pred
    # NOTE: need to take care of possible '\f ->' in predicted types
    # '\\'s are evaluated to one '\' when subing, so we need to add another one
    if "\\" in tp:
        tp = tp.replace("\\", "\\\\")
    llm_prog = re.sub(r"{-@\s*?" + func + r"\s*?::[\s\S]*?@-}", f"{{-@ {func} :: {tp} @-}}", prog, 1)
    # Ignore all other masks and keep the one we want to test for
    for mtype in all_mtypes:
        if all_mtypes[mtype] in llm_prog:
            ignore_func = all_mtypes[mtype].split()[1].strip()
            llm_prog = re.sub(all_mtypes[mtype] + r"\s*?\n", f"{{-@ ignore {ignore_func} @-}}\n", llm_prog, 1)
    # Disable any properties that can't be used yet
    prog_parts = llm_prog.split(f"{{-@ {func} :: {pred} @-}}")
    if "prop_" in prog_parts[1]:
        prog_parts[1] = re.sub(r"{-@\s*?prop_", "{-- prop_", prog_parts[1])
    if "example_" in prog_parts[1]:
        prog_parts[1] = re.sub(r"{-@\s*?example_", "{-- example_", prog_parts[1])
    if "test_" in prog_parts[1]:
        prog_parts[1] = re.sub(r"{-@\s*?test_", "{-- test_", prog_parts[1])
    # Enable any properties that can be used now
    if "prop_" in prog_parts[0]:
        prog_parts[0] = prog_parts[0].replace("{-- prop_", "{-@ prop_")
    if "example_" in prog_parts[0]:
        prog_parts[0] = prog_parts[0].replace("{-- example_", "{-@ example_")
    if "test_" in prog_parts[0]:
        prog_parts[0] = prog_parts[0].replace("{-- test_", "{-@ test_")

    llm_prog = f"{{-@ {func} :: {pred} @-}}".join(prog_parts)
    # if any(used_ground_truth[f] for f in used_ground_truth):
    #     print("-+" * 42)
    #     print(llm_prog)
    #     print("-+" * 42)
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


dependencies = {
    "Ex9.hs": {
        "cons": [],
        "hd": [],
        "tl": [],
        "rot": ["tl", "hd", "cons"],
        "makeq": ["rot"],
        "remove": ["makeq", "tl", "hd"],
        "insert": ["makeq", "cons"],
        "take": ["insert", "remove"]
    },
    "Ex10.hs": {
        "emp": [],
        "lemNotMem": [],
        "mem": ["lemNotMem"],
        "get": ["lemNotMem"],
        "set": [],
        "eval": ["set", "get"],
        "topEval": ["eval"],
        "evalAny": ["eval"],
        "lemNotElem": [],
        "fresh": ["lemNotElem"]
    },
    "Ex11.hs": {
        "create'": [],
        "pack": ["create'"],
        "unpack": [],
        "unsafeTake": [],
        "unsafeDrop": [],
        "chop": ["unsafeTake", "unpack", "pack"],
        "empty": ["pack"],
        "spanByte": ["empty", "unsafeDrop", "unsafeTake"]
    },
    "Ex12.hs": {
        "getHeight": [],
        "mkNode": ["getHeight"],
        "singleton": ["mkNode"],
        "balL0": ["mkNode"],
        "balLL": ["mkNode"],
        "balLR": ["mkNode"],
        "balR0": ["mkNode"],
        "balRR": ["mkNode"],
        "balRL": ["mkNode"],
        "bal": ["balRL", "balRR", "balR0", "balLR", "balLL", "balL0", "mkNode"],
        "insert": ["bal", "singleton"],
        "merge": ["delete", "bal"], # Cycle
        "delete": ["merge", "bal"] # Cycle
    }
}


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

    for target_file in sorted(listdir(path)):
        print(target_file)
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
            ground_truths[func] = clean_type(ground_truth)
            mask_id = int(masked_type.split("<mask_")[1].split(">")[0].strip())
            masked_func_types[mask_id] = masked_type

        masks_per_exer[target_file] = len(masked_func_types)
        type_preds_cache = {}
        tested_types_num = {} # To check locally,
        # how many predictions in this current state we've tested for this type (for backtracking)
        total_times_tested = {} # To check globally,
        # how many predictions we've tested for this type (for stopping backtracking and using ground truth)
        using_ground_truth = {}
        for func in ground_truths:
            key = f"{target_file}--{func}"
            tested_types_num[key] = 0
            total_times_tested[key] = 0
            using_ground_truth[func] = False
        llm_prog = bad_prog
        current_type_state = []
        seen_states = {}
        num_of_iterations = 0
        MAX_ITERATIONS = masks_per_exer[target_file] * 1024 * 2
        mask_id = 1
        while mask_id <= masks_per_exer[target_file]:
            if mask_id == 0:
                print("Backtracked to start; Exiting...", flush=True)
                break
            num_of_iterations += 1
            if num_of_iterations >= MAX_ITERATIONS:
                print(f"Too many iteration {num_of_iterations}; Exiting...", flush=True)

            masked_type = masked_func_types[mask_id]
            func = masked_type.split()[1].strip()
            print("=" * 42)
            print(f"Solving {target_file} ({func})...", flush=True)
            key = f"{target_file}--{func}"
            solved = False
            # NOTE: Exit criteria for the backtracking loop
            # If we have generated too many types for this function
            if key in type_preds_cache and (type_preds_cache[key][1] >= args.max_preds):
                # If we can't generate any good types with LLMs, then test the ground truth (correct type from user)
                if not using_ground_truth[func]:
                    print(f"Testing the ground truth type, since we reached max limit of predictions...", flush=True)
                    type_preds_cache[key] = [ground_truths[func]], args.max_preds + 1
                    using_ground_truth[func] = True
                    for k in tested_types_num:
                        tested_types_num[k] = 0
                    solved = True
                elif all(type_preds_cache[key][1] >= args.max_preds for key in type_preds_cache):
                    # Else just exit the loop
                    print(f"Reached limit of predictions ({type_preds_cache[key][1]} >= {args.max_preds}); Exiting...", flush=True)
                    break
            if not solved and (key not in type_preds_cache or
                               (tested_types_num[key] >= len(type_preds_cache[key][0]) and
                                type_preds_cache[key][1] < args.max_preds)):
                prompt = make_prompt_from_masked_code(llm_prog, mask_id, masked_func_types, target_file)
                mtype_preds = get_type_predictions(prompt, key, func, ground_truths[func], code_llm, args)
                if key in type_preds_cache:
                    preds = type_preds_cache[key][0][:] # It's a list, so we need a deepcopy
                    num_of_preds = type_preds_cache[key][1]
                    preds.extend(mtype_preds)
                    preds = list(set(preds))
                    num_of_preds += args.total_preds
                    # NOTE: Exit criteria for the backtracking loop
                    # If the LLM can't generate any new types, then try the ground truth type or go back
                    if len(preds) == len(type_preds_cache[key][0]):
                        print("No new predictions...", flush=True)
                        if not using_ground_truth[func]:
                            print(f"Testing the ground truth type, since we can't generate any new...", flush=True)
                            preds = [ground_truths[func]]
                            num_of_preds = args.max_preds + 1
                            using_ground_truth[func] = True
                            for k in tested_types_num:
                                tested_types_num[k] = 0
                            solved = True
                        else:
                            llm_prog = restore_mask_at_id(mask_id, func, llm_prog)
                            tested_types_num[key] = 0
                            # We haven't appended a type at the current state stack yet, so no pop() here
                            mask_id -= 1
                            if mask_id > 0:
                                print(f"Backtracking to mask id = {mask_id}...", flush=True)
                                correct_llm_types_per_exer[target_file] -= 1
                                llm_prog = restore_mask_at_id(mask_id, masked_func_types[mask_id].split()[1].strip(), llm_prog)
                            continue
                    type_preds_cache[key] = preds, num_of_preds
                else:
                    type_preds_cache[key] = mtype_preds, args.total_preds

            for type_prediction in type_preds_cache[key][0][tested_types_num[key]:]:
                current_type_state.append(f"{{-@ {func} :: {type_prediction} @-}}")
                tested_types_num[key] += 1
                if "".join(current_type_state) in seen_states:
                    current_type_state.pop()
                    continue
                total_times_tested[key] += 1
                if not using_ground_truth[func] and total_times_tested[key] >= 128:
                    current_type_state.pop()
                    print("Too many failures for this type; Testing the ground truth type...")
                    type_preds_cache[key] = [ground_truths[func]], args.max_preds + 1
                    using_ground_truth[func] = True
                    current_type_state.append(f"{{-@ {func} :: {ground_truths[func]} @-}}")
                    type_prediction = ground_truths[func]
                    for k in tested_types_num:
                        tested_types_num[k] = 0
                print("-" * 42)
                print(f"Testing {{-@ {func} :: {type_prediction} @-}}...", flush=True)
                # NOTE: Just a random check, cause LH crashes for too long types
                if len(type_prediction) > len(ground_truths[func]) + 32:
                    print("...UNSAFE")
                    current_type_state.pop()
                    continue

                llm_prog = replace_type_with_pred(func, type_prediction, llm_prog, masked_func_types)
                if lh_verifies_prog(llm_prog, target_file, args):
                    solved = True
                    seen_states["".join(current_type_state)] = solved
                    print("...SAFE", flush=True)
                    break
                print("...UNSAFE", flush=True)
                seen_states["".join(current_type_state)] = solved
                current_type_state.pop()
                if using_ground_truth[func]:
                    break
            print("-" * 42)
            print("-" * 42)
            if solved:
                print(f"{func} --> SAFE", flush=True)
                correct_llm_types_per_exer[target_file] += 1
                deepest_correct_type_id[target_file] = max(correct_llm_types_per_exer[target_file], deepest_correct_type_id[target_file])
                mask_id += 1
            else:
                print(f"{func} --> UNSAFE", flush=True)
                llm_prog = restore_mask_at_id(mask_id, func, llm_prog)
                mask_id -= 1
                if mask_id > 0:
                    print(f"Backtracking to mask id = {mask_id}...", flush=True)
                    tested_types_num[key] = 0
                    correct_llm_types_per_exer[target_file] -= 1
                    llm_prog = restore_mask_at_id(mask_id, masked_func_types[mask_id].split()[1].strip(), llm_prog)
                elif type_preds_cache[key][1] < args.max_preds:
                    mask_id = 1
                    print(f"Backtracking to mask id = {mask_id}...", flush=True)
                    tested_types_num[key] = len(type_preds_cache[key][0])
            llm_prog = restore_ignored_masks(masked_func_types, llm_prog)

        if correct_llm_types_per_exer[target_file] == masks_per_exer[target_file]:
            fixed_progs += 1
        print("=" * 42)
        print(f"{correct_llm_types_per_exer[target_file]} / {masks_per_exer[target_file]} types predicted correctly")
        print(f"{deepest_correct_type_id[target_file]} / {masks_per_exer[target_file]} deepest type predicted correctly")
        print(tested_types_num)

    print("=" * 42)
    print("=" * 42)
    for k in sorted(correct_llm_types_per_exer.keys()):
        print(f">>> Chapter {k[2:].split('.')[0]}")
        print(f"{correct_llm_types_per_exer[k]} / {masks_per_exer[k]} types predicted correctly")
        print(f"{correct_llm_types_per_exer[k] * 100 / masks_per_exer[k]:.2f}% prediction accuracy")
        print(f"{deepest_correct_type_id[k]} / {masks_per_exer[k]} deepest type predicted correctly")
        print("-" * 42)
    print("=" * 42)
    print("=" * 42)
    print(f"{fixed_progs} / {all_progs} programs fully annotated correctly with LH types")


if __name__ == "__main__":
    cmd_args = get_args()

    run_tests("benchmarks/dependency_tests", cmd_args)