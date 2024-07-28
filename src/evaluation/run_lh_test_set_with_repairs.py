import argparse
from os.path import join
import json
import re
import subprocess as subp
import numpy as np
from collections import defaultdict


MASK_ONLY_POST_COND = False

def get_args():
    parser = argparse.ArgumentParser(description='run_lh_tests')
    parser.add_argument('--cache_file', default="lh_starcoderbase_3B_finetuned_with_the_stack_cache_raw_v3.json",
                        help='use the given file for prompt -> generation cache (default: raw_v3_20_preds cache)')
    parser.add_argument('--data_dir', default="/home/gsakkas/Documents/UCSD/Program-Analytics/liquidhaskell/lh_exercises",
                        help='benchmark data directory (default: liquidhaskell/lh_exercises)')
    parser.add_argument('--print_preds', action="store_true", default=False,
                        help='print the individual type predictions (default: False)')
    parser.add_argument('--out_dir', default="./results",
                        help='output data directory (default: ./results)')
    _args = parser.parse_args()
    return _args


def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n == 0:
        return 0.0
    if c == 0 and n < k:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


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


args = get_args()

diffsToStr = {0: "Easy", 1: "Medium", 2: "Hard"}
difficulties = {
    "Ex3-1.hs--avg": 0,
    "Ex3-2-0.hs--abs": 0,
    "Ex3-3.hs--lAssert": 0,
    "Ex4-2.hs--unsafeLookup": 1,
    "Ex4-5.hs--go'": 0,
    "Ex4-7.hs--absoluteSum'": 0,
    "Ex4-8-1.hs--sparseProduct": 1,
    "Ex4-8.hs--dotProduct": 1,
    "Ex5-0-1.hs--dotProd": 1,
    "Ex5-1.hs--fromList": 1,
    "Ex5-2.hs--plus": 2,
    "Ex5-4.hs--append": 1,
    "Ex6-1-0.hs--divide": 0,
    "Ex6-1.hs--average": 0,
    "Ex6-2.hs--size": 1,
    "Ex6-3-1.hs--head": 0,
    "Ex6-3-2.hs--tail": 0,
    "Ex6-3-3.hs--groupEq": 1,
    "Ex6-3.hs--safeHead": 0,
    "Ex6-4.hs--wtAverage": 1,
    "Ex6-5.hs--risers": 1,
    "Ex7-1.hs--map": 0,
    "Ex7-2.hs--rev'": 1,
    "Ex7-3.hs--zipOrNull": 2,
    "Ex7-4.hs--drop": 1,
    "Ex7-5-1.hs--partition": 0,
    "Ex7-5.hs--take": 2,
    "Ex8-3.hs--reverse'": 1,
    "Ex8-4.hs--halve": 2,
    "Ex8-5.hs--elem": 1,
    "Ex8-8.hs--filter": 2,
    "Ex8-9.hs--rev'": 2,
    "Ex8-10.hs--nub": 0,
    "Ex9-1-0.hs--tl": 1,
    "Ex9-1-1.hs--hd": 1,
    "Ex9-2.hs--rot": 2,
    "Ex9-4-0.hs--makeq": 2,
    "Ex9-4-1.hs--insert": 1,
    "Ex9-6-1.hs--remove": 1,
    "Ex9-6.hs--take": 2,
    "Ex10-1-0.hs--eval": 0,
    "Ex10-1-1.hs--get": 1,
    "Ex10-1-2.hs--lemNotMem": 2,
    "Ex10-1.hs--evalAny": 0,
    "Ex10-3.hs--emp": 1,
    "Ex10-4.hs--set": 1,
    "Ex10-5.hs--mem": 1,
    "Ex10-6.hs--fresh": 1,
    "Ex11-3-0.hs--create'": 0,
    "Ex11-5-0.hs--unsafeTake": 1,
    "Ex11-5-1.hs--unsafeDrop": 1,
    "Ex11-6.hs--go''": 2,
    "Ex11-7.hs--chop": 1,
    "Ex12-1.hs--singleton": 0,
    "Ex12-2-1.hs--balL0": 2,
    "Ex12-2-2.hs--balLL": 2,
    "Ex12-2-3.hs--balLR": 2,
    "Ex12-2.hs--mkNode": 1,
    "Ex12-3.hs--balR0": 2,
    "Ex12-4.hs--balRR": 2,
    "Ex12-5.hs--balRL": 2,
    "Ex12-6-0.hs--insert": 1,
    "Ex12-6-1.hs--insertL": 2,
    "Ex12-6.hs--insertR": 2,
    "Ex12-7-0.hs--bal": 2,
    "Ex12-7-1.hs--insert'": 1,
    "Ex12-7-2.hs--delete": 1,
    "Ex12-7-3.hs--merge": 2
}

with open(args.cache_file, "r", encoding="utf-8") as cache_fin:
    cache = json.loads(cache_fin.read())

fixed_progs = 0
all_progs = 0
samples_per_difficulty = {k: 0 for k in [0, 1, 2]}
llm_preds_per_difficulty = {k: 0 for k in [0, 1, 2]}
samples_per_func = defaultdict(int)
llm_preds_per_func = defaultdict(int)
samples_per_exer = defaultdict(int)
llm_preds_per_exer = defaultdict(int)
func_pass_at_1 = {}
func_pass_at_10 = {}
func_pass_at_50 = {}
path_to_testset = "../liquidhaskell/lh_exercises/correct_baselines"
for key in sorted(cache.keys()):
    all_progs += 1
    func = key.split("--")[-1]
    exercise = key.split("-")[0]
    print(f"Solving {key.split('--')[0]} ({func})...", flush=True)
    samples_per_difficulty[difficulties[key]] += 1
    samples_per_func[key] += len(cache[key])
    samples_per_exer[exercise] += 1
    bfile = key.split("--")[0].split(".hs")[0].split("/")[-1].replace("flycheck_", "")
    fix_prog = ""
    with open(join(path_to_testset, bfile + "_correct.hs"), "r", encoding="utf-8") as good_fin:
        fix_prog = good_fin.read()
    ground_truth = re.findall(r"{-@\s*?" + func + r"\s*?::([\s\S]*?)@-}", fix_prog)
    solved = False
    seen_preds = {}
    for llm_pred in cache[key]:
        with open(join(args.data_dir, bfile + "_llm.hs"), "w", encoding="utf-8") as llm_fin:
            if "@-}" in llm_pred:
                llm_pred = llm_pred.split("@-}")[0].rstrip()
            if "\\" in llm_pred:
                llm_pred = llm_pred.replace("\\", "\\\\")
            if "<file_sep>" in llm_pred:
                llm_pred = llm_pred.split("<file_sep>")[0]
            if llm_pred in seen_preds:
                if seen_preds[llm_pred]:
                    llm_preds_per_func[key] += 1
                continue
            if args.print_preds:
                print("--------------------------------------")
                print(f"{{-@ {func} :: {llm_pred} @-}}")
                print("--------------------------------------")

            llm_prog = fix_prog
            if MASK_ONLY_POST_COND:
                return_type = ground_truth[0].split("->")[-1].strip()
                pre_cond = ' -> '.join([typ.strip() for typ in ground_truth[0].split("->")[:-1]])
                llm_prog = re.sub(r"{-@\s*?" + func + r"\s*?::[\s\S]*?@-}", f"{{-@ {func} :: {pre_cond} -> {llm_pred} @-}}", fix_prog, 1)
            else:
                llm_prog = re.sub(r"{-@\s*?" + func + r"\s*?::[\s\S]*?@-}", f"{{-@ {func} :: {llm_pred} @-}}", fix_prog, 1)
            llm_fin.write(llm_prog)
            # print(llm_prog, flush=True)

        # TODO: Just a random check, cause LH crashes for too long types
        if len(llm_pred) > len(ground_truth[0]) + 64:
            seen_preds[llm_pred] = False
            if args.print_preds:
                print("UNSAFE", flush=True)
        else:
            cmds = "source /home/gsakkas/.ghcup/env; " # for local Haskell installation
            cmds += "export PATH=$PATH:/home/gsakkas/usr/bin; " # for local Z3 installation
            cmds += f"cd {args.data_dir}; "
            cmds += f"rm {bfile}_llm.hi; "
            cmds += f"stack exec ghc -- -fplugin=LiquidHaskell {bfile}_llm.hs"
            test_output = subp.run(cmds, shell=True, check=False, capture_output=True)
            result = test_output.stdout.decode('utf-8').strip()
            if result != "" and "UNSAFE" not in result and "SAFE" in result:
                seen_preds[llm_pred] = solved = True
                llm_preds_per_func[key] += 1
                if args.print_preds:
                    print("SAFE", flush=True)
            else:
                seen_preds[llm_pred] = False
                if args.print_preds:
                    print("UNSAFE", flush=True)
    print(f"{len(seen_preds)} unique preds", flush=True)
    if solved:
        fixed_progs += 1
        llm_preds_per_difficulty[difficulties[key]] += 1
        llm_preds_per_exer[exercise] += 1
        print("--> SAFE")
    else:
        print("--> UNSAFE")
    func_pass_at_1[key] = pass_at_k(samples_per_func[key], llm_preds_per_func[key], 1) * 100
    func_pass_at_10[key] = pass_at_k(samples_per_func[key], llm_preds_per_func[key], 10) * 100
    func_pass_at_50[key] = pass_at_k(samples_per_func[key], llm_preds_per_func[key], 50) * 100
    print(f"  - pass@1  = {func_pass_at_1[key]:.2f}")
    print(f"  - pass@10 = {func_pass_at_10[key]:.2f}")
    print(f"  - pass@50 = {func_pass_at_50[key]:.2f}")
    print(f"{llm_preds_per_func[key]} / {samples_per_func[key]} llm type predictions")
    print(f"{llm_preds_per_func[key] * 100 / samples_per_func[key]:.2f}% func predictions accuracy")
    print("--------------------------------------", flush=True)
    print("--------------------------------------", flush=True)
    print("--------------------------------------", flush=True)


print("============================================================")
print("============================================================")
print(f"{fixed_progs} / {all_progs} llm type predictions")
print(f"{fixed_progs * 100 / all_progs:.2f}% llm accuracy")
print(f"avg. pass@1  = {sum(func_pass_at_1.values()) / len(func_pass_at_1):.2f}% llm accuracy")
print(f"avg. pass@10  = {sum(func_pass_at_10.values()) / len(func_pass_at_10):.2f}% llm accuracy")
print(f"avg. pass@50  = {sum(func_pass_at_50.values()) / len(func_pass_at_50):.2f}% llm accuracy")
print("============================================================")
print("============================================================")
for key in [0, 1, 2]:
    print("------------------------------------------------------------")
    print(f">>> {diffsToStr[key]} difficulty")
    print(f"{llm_preds_per_difficulty[key]} / {samples_per_difficulty[key]} llm type predictions")
    print(f"{llm_preds_per_difficulty[key] * 100 / samples_per_difficulty[key]:.2f}% accuracy")
print("============================================================")
print("============================================================")
for key in sorted(llm_preds_per_exer.keys()):
    print("------------------------------------------------------------")
    print(f">>> Chapter {key[2:]}")
    print(f"{llm_preds_per_exer[key]} / {samples_per_exer[key]} llm type predictions")
    print(f"{llm_preds_per_exer[key] * 100 / samples_per_exer[key]:.2f}% accuracy")