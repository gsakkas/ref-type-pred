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
    # print_prog_liquid_types(one_mask_bad_prog)
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


def flip_properties(prog, curr_type):
    # Disable any properties that can't be used yet
    prog_parts = prog.split(curr_type)
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

    llm_prog = curr_type.join(prog_parts)
    return llm_prog


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


def get_type_state_str(state):
    state_str = ""
    for func, ftype in state.items():
        if ftype:
            state_str += f"{func} :: {ftype}<->"
    return state_str[:-3]


def print_prog_liquid_types(prog):
    types_str = ""
    in_type = False
    for l in prog.split('\n'):
        if "{-@" in l or "{--" in l:
            in_type = True
        if in_type:
            types_str += l + "\n"
        if "@-}" in l:
            in_type = False
    print("-" * 42 + "Types in program" + "-" * 42)
    print(types_str)
    print("-" * 100)


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
        # "merge": ["delete", "bal"], # Cycle
        # "delete": ["merge", "bal"] # Cycle
        "delete": ["bal"] # Cycle
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
    exit_mask_id = {k: 0 for k in set(difficulties.keys())}
    deepest_correct_type_id = {k: 0 for k in set(difficulties.keys())}
    total_ground_truths = {k: 0 for k in set(difficulties.keys())}

    # Load (finetuned) code LLM
    code_llm = StarCoderModel()

    for target_file in sorted(listdir(path)):
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
        func_to_mask_id = {}
        for masked_type in masked_funcs:
            func = masked_type.split()[1].strip()
            ground_truth = re.findall(r"{-@\s*?" + func + r"\s*?::([\s\S]*?)@-}", fix_prog)[0].strip()
            ground_truths[func] = clean_type(ground_truth)
            mask_id = int(masked_type.split("<mask_")[1].split(">")[0].strip())
            masked_func_types[mask_id] = masked_type
            func_to_mask_id[func] = mask_id

        masks_per_exer[target_file] = len(masked_func_types)
        type_preds_cache = {}
        tested_types_num = {} # To check locally,
        # how many predictions in this current state we've tested for this type (for backtracking)
        total_times_tested = {} # To check globally,
        # how many predictions we've tested for this type (for stopping backtracking and using ground truth)
        using_ground_truth = {}
        current_type_state = {}
        for func in ground_truths:
            key = f"{target_file}--{func}"
            tested_types_num[func] = 0
            total_times_tested[func] = 0
            using_ground_truth[func] = False
            current_type_state[func] = ""
        llm_prog = bad_prog
        seen_states = {}
        num_of_iterations = 0
        MAX_ITERATIONS = masks_per_exer[target_file] * 50
        mask_stack = [i for i in range(len(masked_func_types), 0, -1)]
        mask_id = 1
        while mask_stack:
            mask_id = mask_stack.pop()
            exit_mask_id[target_file] = mask_id
            if mask_id == 0:
                print("Backtracked to start; Exiting...", flush=True)
                break
            # for mid in reversed(mask_stack):
            #     temp_1 = masked_func_types[mid]
            #     temp_2 = temp_1.split()[1].strip()
            #     print(f"|{temp_2}|")
            num_of_iterations += 1
            if num_of_iterations >= MAX_ITERATIONS:
                print(f"Too many iterations {num_of_iterations}; Exiting...", flush=True)
                break

            masked_type = masked_func_types[mask_id]
            func = masked_type.split()[1].strip()
            print("=" * 42)
            print(f"Solving {target_file} ({func})...", flush=True)
            key = f"{target_file}--{func}"
            solved = False
            llm_prog = restore_ignored_masks(masked_func_types, llm_prog)
            # NOTE: Exit criteria for the backtracking loop
            # If we have generated too many types for this function
            if key in type_preds_cache and type_preds_cache[key][1] >= args.max_preds:
                # If we can't generate any good types with LLMs, then test the ground truth (correct type from user)
                if not using_ground_truth[func]:
                    print(f"Testing the ground truth type, since we reached max limit of predictions...", flush=True)
                    type_preds_cache[key] = [ground_truths[func]], args.max_preds + 1
                    using_ground_truth[func] = True
                    tested_types_num[func] = 0
                    for func_dep in dependencies[target_file][func]:
                        tested_types_num[func_dep] = 0
                    solved = True
                elif all(type_preds_cache[key][1] >= args.max_preds for key in type_preds_cache):
                    # Else just exit the loop
                    print(f"Reached limit of predictions ({type_preds_cache[key][1]} >= {args.max_preds}); Exiting...", flush=True)
                    break
            if not using_ground_truth[func] and \
                (key not in type_preds_cache or
                (tested_types_num[func] >= len(type_preds_cache[key][0]) and
                    type_preds_cache[key][1] < args.max_preds)):
                prompt = make_prompt_from_masked_code(llm_prog, mask_id, masked_func_types, target_file)
                mtype_preds = get_type_predictions(prompt, key, func, ground_truths[func], code_llm, args)
                num_of_preds = args.total_preds
                if key in type_preds_cache:
                    prev_preds = type_preds_cache[key][0]
                    mtype_preds.extend(prev_preds)
                    mtype_preds = list(set(mtype_preds))
                    num_of_preds += type_preds_cache[key][1]
                    # NOTE: Exit criteria for the backtracking loop
                    # If the LLM can't generate any new types, then try the ground truth type or go back
                    # NOTE: Or if LLM generates too many different types (heuristic), probably they are not that good
                    if len(mtype_preds) == len(type_preds_cache[key][0]) and dependencies[target_file][func]:
                        print("No new predictions...", flush=True)
                        # Clean up all function depending on this
                        for func_next, deps in reversed(dependencies[target_file].items()):
                            if func in deps and func_to_mask_id[func_next] not in mask_stack:
                                mask_stack.append(func_to_mask_id[func_next])
                                llm_prog = restore_mask_at_id(func_to_mask_id[func_next], func_next, llm_prog)
                                tested_types_num[func_next] = 0
                                current_type_state[func_next] = ""
                        # Add failed mask_id to retry later
                        mask_stack.append(mask_id)
                        llm_prog = restore_mask_at_id(mask_id, func, llm_prog)
                        current_type_state[func] = ""
                        tested_types_num[func] = 0
                        # Add least tested dependency to stack
                        next_func = dependencies[target_file][func][0]
                        untested_types = len(type_preds_cache[f"{target_file}--{next_func}"][0]) - tested_types_num[next_func]
                        next_mask_id = func_to_mask_id[next_func]
                        for func_dep in dependencies[target_file][func]:
                            if len(type_preds_cache[f"{target_file}--{func_dep}"][0]) - tested_types_num[func_dep] > untested_types:
                                next_mask_id = func_to_mask_id[func_dep]
                                untested_types = len(type_preds_cache[f"{target_file}--{func_dep}"][0]) - tested_types_num[func_dep]
                                next_func = func_dep
                        mask_stack.append(next_mask_id)
                        llm_prog = restore_mask_at_id(next_mask_id, next_func, llm_prog)
                        current_type_state[next_func] = ""
                        print(f"Backtracking to mask id = {next_mask_id}...", flush=True)
                        continue
                    elif len(mtype_preds) == len(type_preds_cache[key][0]) and mask_id > 1:
                        # Clean up all function depending on this
                        for func_next, deps in reversed(dependencies[target_file].items()):
                            if func in deps and func_to_mask_id[func_next] not in mask_stack:
                                mask_stack.append(func_to_mask_id[func_next])
                                llm_prog = restore_mask_at_id(func_to_mask_id[func_next], func_next, llm_prog)
                                tested_types_num[func_next] = 0
                                current_type_state[func_next] = ""
                        # Add failed mask_id to retry later
                        mask_stack.append(mask_id)
                        llm_prog = restore_mask_at_id(mask_id, func, llm_prog)
                        current_type_state[func] = ""
                        tested_types_num[func] = 0
                        # Add least tested dependency to stack
                        mask_id -= 1
                        next_func = masked_func_types[mask_id].split()[1].strip()
                        mask_stack.append(mask_id)
                        llm_prog = restore_mask_at_id(mask_id, next_func, llm_prog)
                        current_type_state[next_func] = ""
                        print(f"Backtracking to mask id = {mask_id}...", flush=True)
                        continue
                    elif (len(mtype_preds) > 10 or len(mtype_preds) == len(type_preds_cache[key][0])) and not using_ground_truth[func]:
                        print(f"Testing the ground truth type, since we got too many unique predictions...", flush=True)
                        mtype_preds = [ground_truths[func]]
                        num_of_preds = args.max_preds + 1
                        using_ground_truth[func] = True
                        tested_types_num[func] = 0
                        for func_dep in dependencies[target_file][func]:
                            tested_types_num[func_dep] = 0
                        solved = True
                elif len(mtype_preds) < 5:
                    mtype_preds.extend(get_type_predictions(prompt, key, func, ground_truths[func], code_llm, args))
                    mtype_preds = list(set(mtype_preds))
                    num_of_preds += args.total_preds
                type_preds_cache[key] = mtype_preds, num_of_preds

            if using_ground_truth[func]:
                current_type_state[func] = ground_truths[func]
                total_times_tested[func] += 1
                print("-" * 42)
                print(f"Testing {{-@ {func} :: {ground_truths[func]} @-}}...", flush=True)
                # for f, t in current_type_state.items():
                #     if t:
                #         print(f">>> {f} :: {t}")
                state = get_type_state_str(current_type_state)
                llm_prog = replace_type_with_pred(func, ground_truths[func], llm_prog, masked_func_types)
                # print_prog_liquid_types(llm_prog)
                prev_ft = None
                for f, t in current_type_state.items():
                    if not t:
                        break
                    prev_ft = f"{{-@ {f} :: {t} @-}}"
                # print(prev_ft)
                llm_prog = flip_properties(llm_prog, prev_ft)
                if state in seen_states:
                    print("Tested before.....")
                    solved = seen_states[state]
                else:
                    if lh_verifies_prog(llm_prog, target_file, args):
                        seen_states[state] = solved = True
                        print("...SAFE", flush=True)
                    else:
                        print("...UNSAFE", flush=True)
                        seen_states[state] = solved = False
            else:
                for type_prediction in type_preds_cache[key][0][tested_types_num[func]:]:
                    current_type_state[func] = type_prediction
                    tested_types_num[func] += 1
                    total_times_tested[func] += 1
                    print("-" * 42)
                    print(f"Testing {{-@ {func} :: {type_prediction} @-}}...", flush=True)
                    # NOTE: Just a random check, cause LH crashes for too long types
                    if len(type_prediction) > len(ground_truths[func]) + 32:
                        print("...UNSAFE")
                        current_type_state[func] = ""
                        continue
                    if not using_ground_truth[func] and total_times_tested[func] >= 25:
                        print("Too many failures for this type; Testing the ground truth type...")
                        type_preds_cache[key] = [ground_truths[func]], args.max_preds + 1
                        using_ground_truth[func] = True
                        current_type_state[func] = ground_truths[func]
                        type_prediction = ground_truths[func]
                        tested_types_num[func] = 0
                        for func_dep in dependencies[target_file][func]:
                            tested_types_num[func_dep] = 0
                        print(f"Testing {{-@ {func} :: {type_prediction} @-}}...", flush=True)
                    # for f, t in current_type_state.items():
                    #     if t:
                    #         print(f">>> {f} :: {t}")
                    state = get_type_state_str(current_type_state)
                    llm_prog = replace_type_with_pred(func, type_prediction, llm_prog, masked_func_types)
                    # print_prog_liquid_types(llm_prog)
                    prev_ft = None
                    for f, t in current_type_state.items():
                        if not t:
                            break
                        prev_ft = f"{{-@ {f} :: {t} @-}}"
                    # print(prev_ft)
                    llm_prog = flip_properties(llm_prog, prev_ft)
                    if state in seen_states and not seen_states[state]:
                        print("Tested before.....")
                        continue
                    if state in seen_states and seen_states[state]:
                        solved = True
                        break
                    if lh_verifies_prog(llm_prog, target_file, args):
                        seen_states[state] = solved = True
                        print("...SAFE", flush=True)
                        break
                    print("...UNSAFE", flush=True)
                    seen_states[state] = solved = False
            print("-" * 42)
            print("-" * 42)
            if solved:
                print(f"{func} --> SAFE", flush=True)
                deepest_correct_type_id[target_file] = max(mask_id, deepest_correct_type_id[target_file])
            else:
                print(f"{func} --> UNSAFE", flush=True)
                # if using_ground_truth[func] and dependencies[target_file][func]:
                #     print("Something wrong with dependencies while using ground truth...")
                #     # Clean up all function depending on this
                #     for func_next, deps in reversed(dependencies[target_file].items()):
                #         if func in deps and func_to_mask_id[func_next] not in mask_stack:
                #             mask_stack.append(func_to_mask_id[func_next])
                #             llm_prog = restore_mask_at_id(func_to_mask_id[func_next], func_next, llm_prog)
                #             tested_types_num[func_next] = 0
                #             current_type_state[func_next] = ""
                #     # Add failed mask_id to retry later
                #     mask_stack.append(mask_id)
                #     llm_prog = restore_mask_at_id(mask_id, func, llm_prog)
                #     current_type_state[func] = ""
                #     # Add all dependencies to stack
                #     # And clean up dependencies again... FIXME: is this necessearry?
                #     for func_dep in dependencies[target_file][func]:
                #         mask_stack.append(func_to_mask_id[func_dep])
                #         llm_prog = restore_mask_at_id(func_to_mask_id[func_dep], func_dep, llm_prog)
                #         tested_types_num[func_dep] = 0
                #         current_type_state[func_dep] = ""
                #     print(f"Backtracking to mask id = {mask_stack[-1]}...", flush=True)
                if dependencies[target_file][func]:
                    # Clean up all function depending on this
                    for func_next, deps in reversed(dependencies[target_file].items()):
                        if func in deps and func_to_mask_id[func_next] not in mask_stack:
                            mask_stack.append(func_to_mask_id[func_next])
                            llm_prog = restore_mask_at_id(func_to_mask_id[func_next], func_next, llm_prog)
                            tested_types_num[func_next] = 0
                            current_type_state[func_next] = ""
                    # Add failed mask_id to retry later
                    mask_stack.append(mask_id)
                    llm_prog = restore_mask_at_id(mask_id, func, llm_prog)
                    current_type_state[func] = ""
                    tested_types_num[func] = 0
                    # Add least tested dependency to stack
                    next_func = dependencies[target_file][func][0]
                    untested_types = len(type_preds_cache[f"{target_file}--{next_func}"][0]) - tested_types_num[next_func]
                    next_mask_id = func_to_mask_id[next_func]
                    for func_dep in dependencies[target_file][func]:
                        if len(type_preds_cache[f"{target_file}--{func_dep}"][0]) - tested_types_num[func_dep] > untested_types:
                            next_mask_id = func_to_mask_id[func_dep]
                            untested_types = len(type_preds_cache[f"{target_file}--{func_dep}"][0]) - tested_types_num[func_dep]
                            next_func = func_dep
                    mask_stack.append(next_mask_id)
                    llm_prog = restore_mask_at_id(next_mask_id, next_func, llm_prog)
                    current_type_state[next_func] = ""
                    print(f"Backtracking to mask id = {next_mask_id}...", flush=True)
                elif not using_ground_truth[func]:
                    # Add failed mask_id to retry later
                    mask_stack.append(mask_id)
                    llm_prog = restore_mask_at_id(mask_id, func, llm_prog)
                    current_type_state[func] = ""
                    print(f"Trying again mask id = {mask_id}...", flush=True)
                else:
                    tested_types_num[func] = 0
                    mask_stack.append(mask_id-1)
                    print(f"Backtracking to previous mask id = {mask_id-1}...", flush=True)

        if exit_mask_id[target_file] == masks_per_exer[target_file]:
            fixed_progs += 1
        print("=" * 42)
        print(f"{exit_mask_id[target_file]} / {masks_per_exer[target_file]} exit location")
        print(f"{deepest_correct_type_id[target_file]} / {masks_per_exer[target_file]} types predicted correctly")
        total_ground_truths[target_file] = 0
        for k in using_ground_truth:
            if using_ground_truth[k]:
                total_ground_truths[target_file] += 1
        print(f"{total_ground_truths[target_file]} ground truth types used")

    print("=" * 42)
    print("=" * 42)
    for k in sorted(exit_mask_id.keys()):
        if masks_per_exer[k] > 0:
            print(f">>> Chapter {k[2:].split('.')[0]}")
            print(f"{exit_mask_id[k]} / {masks_per_exer[k]} exit location")
            print(f"{deepest_correct_type_id[k]} / {masks_per_exer[k]} types predicted correctly")
            print(f"{deepest_correct_type_id[k] * 100 / masks_per_exer[k]:.2f}% prediction accuracy")
            print(f"{total_ground_truths[k]} ground truth types used")
            print("-" * 42)
    print("=" * 42)
    print("=" * 42)
    print(f"{fixed_progs} / {all_progs} programs fully annotated correctly with LH types")


if __name__ == "__main__":
    cmd_args = get_args()

    run_tests("benchmarks/dependency_tests", cmd_args)