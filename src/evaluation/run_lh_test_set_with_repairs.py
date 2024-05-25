import argparse
from os.path import join
import json
import re
import subprocess as subp


KEEP_AND_COMMENT_NON_CODE = False
USE_BUGGY_LINES_AS_COMMENT = False
MASK_ONLY_POST_COND = False

def get_args():
    parser = argparse.ArgumentParser(description='run_lh_tests')
    parser.add_argument('--cache_file', default="lh_starcoderbase_3B_finetuned_with_the_stack_cache_raw_v3.json",
                        help='use the given file for prompt -> generation cache (default: raw_v3_20_repairs cache)')
    parser.add_argument('--data_dir', default="/home/gsakkas/Documents/UCSD/Program-Analytics/liquidhaskell/lh_exercises",
                        help='benchmark data directory (default: liquidhaskell/lh_exercises)')
    parser.add_argument('--print_preds', action="store_true", default=False,
                        help='print the individual type predictions (default: False)')
    parser.add_argument('--out_dir', default="./results",
                        help='output data directory (default: ./results)')
    _args = parser.parse_args()
    return _args


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

# cache_file = "lh_starcoderbase_3B_finetuned_with_the_stack_chkpnt_20000_cache_raw_v3.json"
# cache_file = "lh_starcoderbase_3B_finetuned_with_the_stack_chkpnt_20000_cache_with_haskell_types_v3.json"
# cache_file = "lh_starcoderbase_3B_finetuned_with_the_stack_chkpnt_20000_cache_with_haskell_types_and_tests_v3_temp_098.json"

with open(args.cache_file, "r", encoding="utf-8") as cache_fin:
    cache = json.loads(cache_fin.read())

fixed_progs = 0
all_progs = 0
samples_per_difficulty = {k: 0 for k in [0, 1, 2]}
llm_repairs_per_difficulty = {k: 0 for k in [0, 1, 2]}
samples_per_func = {k.split("--")[-1]: 0 for k in set(cache.keys())}
llm_repairs_per_func = {k.split("--")[-1]: 0 for k in set(cache.keys())}
samples_per_exer = {k.split("-")[0]: 0 for k in set(cache.keys())}
llm_repairs_per_exer = {k.split("-")[0]: 0 for k in set(cache.keys())}
path_to_testset = "../liquidhaskell/lh_exercises/correct_baselines"
for k in list(cache.keys()):
    all_progs += 1
    func = k.split("--")[-1]
    exercise = k.split("-")[0]
    print(f"Solving {k.split('--')[0]} ({func})...")
    samples_per_difficulty[difficulties[k]] += 1
    samples_per_func[func] += 1
    samples_per_exer[exercise] += 1
    bfile = k.split("--")[0].split(".hs")[0].split("/")[-1].replace("flycheck_", "")
    fix_prog = ""
    with open(join(path_to_testset, bfile + "_correct.hs"), "r", encoding="utf-8") as good_fin:
        fix_prog = good_fin.read()
    seen_repairs = set()
    for llm_repair in cache[k]:
        with open(join(path_to_testset, bfile + "_llm.hs"), "w", encoding="utf-8") as llm_fin:
            if "@-}" in llm_repair:
                llm_repair = llm_repair.split("@-}")[0].rstrip()
            if "\\" in llm_repair:
                llm_repair = llm_repair.replace("\\", "\\\\")
            if "<file_sep>" in llm_repair:
                llm_repair = llm_repair.split("<file_sep>")[0]
            if llm_repair in seen_repairs:
                continue
            seen_repairs.add(llm_repair)
            if args.print_preds:
                print("--------------------------------------")
                print(f"{{-@ {func} :: {llm_repair} @-}}")
                print("--------------------------------------")

            ground_truth = re.findall(r"{-@\s*?" + func + r"\s*?::([\s\S]*?)@-}", fix_prog)
            # TODO: Just a random check, cause LH crashes for too long types
            if len(llm_repair) > len(ground_truth[0]) + 32:
                if args.print_preds:
                    print("UNSAFE")
                continue

            llm_prog = fix_prog
            if MASK_ONLY_POST_COND:
                return_type = ground_truth[0].split("->")[-1].strip()
                pre_cond = ' -> '.join([typ.strip() for typ in ground_truth[0].split("->")[:-1]])
                llm_prog = re.sub(r"{-@\s*?" + func + r"\s*?::[\s\S]*?@-}", f"{{-@ {func} :: {pre_cond} -> {llm_repair} @-}}", fix_prog, 1)
            else:
                llm_prog = re.sub(r"{-@\s*?" + func + r"\s*?::[\s\S]*?@-}", f"{{-@ {func} :: {llm_repair} @-}}", fix_prog, 1)
            llm_fin.write(llm_prog)
            # print(llm_prog, flush=True)

        cmds = "source /home/gsakkas/.ghcup/env; " # for local Haskell installation
        cmds += "export PATH=$PATH:/home/gsakkas/usr/bin; " # for local Z3 installation
        cmds += f"cd {args.data_dir}; "
        cmds += f"rm {bfile}_llm.hi; "
        cmds += f"stack exec ghc -- -fplugin=LiquidHaskell {bfile}_llm.hs"
        test_output = subp.run(cmds, shell=True, check=False, capture_output=True)
        result = test_output.stdout.decode('utf-8').strip()
        if result != "" and "UNSAFE" not in result and "SAFE" in result:
            fixed_progs += 1
            llm_repairs_per_difficulty[difficulties[k]] += 1
            llm_repairs_per_func[func] += 1
            llm_repairs_per_exer[exercise] += 1
            if args.print_preds:
                print("SAFE", flush=True)
            break
        if args.print_preds:
            print("UNSAFE", flush=True)
    print(f"{len(seen_repairs)} unique repairs", flush=True)
    if llm_repairs_per_func[func] > 0:
        print("--> SAFE")
    else:
        print("--> UNSAFE")
    print("--------------------------------------", flush=True)


print("============================================================")
print("============================================================")
print(f"{fixed_progs} / {all_progs} llm repairs")
print(f"{fixed_progs * 100 / all_progs:.2f}% llm repair rate")
print("============================================================")
print("============================================================")
for k in [0, 1, 2]:
    print("------------------------------------------------------------")
    print(f">>> {diffsToStr[k]} difficulty")
    print(f"{llm_repairs_per_difficulty[k]} / {samples_per_difficulty[k]} llm repairs")
    print(f"{llm_repairs_per_difficulty[k] * 100 / samples_per_difficulty[k]:.2f}% repair rate")
print("============================================================")
print("============================================================")
for k in sorted(llm_repairs_per_func.keys()):
    print("------------------------------------------------------------")
    print(f">>> func == `{k}`")
    print(f"{llm_repairs_per_func[k]} / {samples_per_func[k]} llm repairs")
    print(f"{llm_repairs_per_func[k] * 100 / samples_per_func[k]:.2f}% repair rate")
print("============================================================")
print("============================================================")
for k in sorted(llm_repairs_per_exer.keys()):
    print("------------------------------------------------------------")
    print(f">>> Chapter {k[2:]}")
    print(f"{llm_repairs_per_exer[k]} / {samples_per_exer[k]} llm repairs")
    print(f"{llm_repairs_per_exer[k] * 100 / samples_per_exer[k]:.2f}% repair rate")