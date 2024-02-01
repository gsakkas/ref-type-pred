import argparse
from os.path import exists, join
from os import listdir
import json
import re
import subprocess as subp
from src.predict.get_starcoder_code_suggestions import get_starcoder_code_suggestions

def get_args():
    parser = argparse.ArgumentParser(description='run_dependency_lh_tests')
    parser.add_argument('--total_repairs', default=10, type=int,
                        help='total repairs to generate with the model (default: 10)')
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


def make_prompt_from_masked_code(badp, fixp, masked_types, filename):
    FIM_PREFIX = "<fim_prefix>"
    FIM_MIDDLE = "<fim_middle>"
    FIM_SUFFIX = "<fim_suffix>"

    correct_types = {}
    for masked_type in masked_types:
        func = masked_type.split()[1].strip()
        ground_truth = re.findall(r"{-@\s*?" + func + r"\s*?::([\s\S]*?)@-}", fixp)[0].strip()
        if not ground_truth:
            continue
        correct_types[func] = ground_truth

    prompts = []
    for mask_id in range(1, len(masked_types)+1):
        prefix = f"<filename>solutions/{filename}\n-- Fill in the masked refinement type in the following LiquidHaskell program\n"
        split_code = badp.split(f"<mask_{mask_id}>")
        prompt = f"{FIM_PREFIX}{prefix}{split_code[0]}{FIM_SUFFIX}{split_code[1]}{FIM_MIDDLE}"
        prompts.append(prompt)

    return prompts, correct_types


def get_type_predictions(masked_types, all_prompts, filename, args):
    cache_file = join(args.out_dir, args.cache_file)
    cache = {}
    if (args.use_cache or args.update_cache or args.create_cache_only) and exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as cf:
            cache = json.loads(cf.read())

    if args.print_logs:
        print("-" * 42)
    for mtype, prompt in zip(masked_types, all_prompts):
        if args.print_logs:
            print(f"-> {mtype}")
        func = mtype.split()[1].strip()
        key = f"{filename}--{func}"
        if args.use_cache and key in cache and cache[key] != []:
            prog_repairs = cache[key]
        else:
            prog_repairs = get_starcoder_code_suggestions(prompt, args.total_repairs)
            prog_repairs = list(set(prog_repairs))
            if args.print_logs:
                print(f"{len(prog_repairs)} unique predicted types")
        if args.update_cache or args.create_cache_only:
            cache[key] = prog_repairs

    if args.update_cache or args.create_cache_only:
        with open(cache_file, "w", encoding="utf-8") as cf:
            cf.write(json.dumps(cache, indent = 4))

    return cache


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
        masked_func_types = list(filter(lambda x: "<mask_" in x, all_liquid_types))
        all_prompts, ground_truths = make_prompt_from_masked_code(bad_prog, fix_prog, masked_func_types, target_file)
        type_preds_cache = get_type_predictions(masked_func_types, all_prompts, target_file, args)
        if args.create_cache_only:
            continue

        masks_per_exer[target_file] = len(masked_func_types)
        for masked_type in masked_func_types:
            func = masked_type.split()[1].strip()
            print(f"Solving {target_file} ({func})...")
            key = f"{target_file}--{func}"
            solved = False
            for type_prediction in type_preds_cache[key]:
                with open(join(args.exec_dir, target_file.replace(".hs", "_llm.hs")), "w", encoding="utf-8") as llm_fin:
                    if "@-}" in type_prediction:
                        type_prediction = type_prediction.split("@-}")[0].rstrip()
                    if "\\" in type_prediction:
                        type_prediction = type_prediction.replace("\\", "\\\\")
                    if args.print_logs:
                        print("-" * 42)
                        print(f"{{-@ {func} :: {type_prediction} @-}}")
                        print("-" * 42)

                    # NOTE: Just a random check, cause LH crashes for too long types
                    if len(type_prediction) > len(ground_truths[func]) + 32:
                        if args.print_logs:
                            print("UNSAFE")
                        continue

                    llm_prog = re.sub(r"{-@\s*?" + func + r"\s*?::[\s\S]*?@-}", f"{{-@ {func} :: {type_prediction} @-}}", fix_prog, 1)
                    llm_fin.write(llm_prog)

                cmds = f"cd {args.exec_dir}; "
                cmds += f"stack exec ghc -- -fplugin=LiquidHaskell {target_file.replace('.hs', '_llm.hs')}"
                test_output = subp.run(cmds, shell=True, check=False, capture_output=True)
                result = test_output.stdout.decode('utf-8').strip()
                if result != "" and "UNSAFE" not in result and "SAFE" in result:
                    correct_llm_types_per_exer[target_file] += 1
                    solved = True
                    if args.print_logs:
                        print("SAFE")
                    break
                if args.print_logs:
                    print("UNSAFE")
            if solved:
                print("--> SAFE")
            else:
                print("--> UNSAFE")

        if correct_llm_types_per_exer[target_file] == masks_per_exer[target_file]:
            fixed_progs += 1
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