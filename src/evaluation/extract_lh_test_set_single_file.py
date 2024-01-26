import json
from prog_diff import get_masked_lines

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

# The prompt that the StarCoder paper claims to improve the results
starcoder_prompt = "<filename>solutions/{file_name}\n-- Fill the refinement type errors in the following LiquidHaskell code exercise\n"
json_dataset = []
ground_truth_types = {}
sample_pair = {}
sample_pair["bad"] = """
import Data.Set (Set)
import qualified Data.Set as S
data LL a = Nil | Cons a (LL a)

{-@ measure elems @-}
elems :: LL a -> Set a
elems Nil = S.empty
elems (Cons h l) = S.union (S.singleton h) (elems l)

{-@ rev :: xs:LL a -> {v:LL a | elems v == elems xs} @-}
rev :: LL a -> LL a
rev l = helper Nil l

{-@ helper :: ??? @-}
helper :: LL a -> LL a -> LL a
helper acc Nil = acc
helper acc (Cons h l) = helper (Cons h acc) l
"""

sample_pair["fix"] = """
import Data.Set (Set)
import qualified Data.Set as S
data LL a = Nil | Cons a (LL a)

{-@ measure elems @-}
elems :: LL a -> Set a
elems Nil = S.empty
elems (Cons h l) = S.union (S.singleton h) (elems l)

{-@ rev :: xs:LL a -> {v:LL a | elems v == elems xs} @-}
rev :: LL a -> LL a
rev l = helper Nil l

{-@ helper :: xs:LL a -> ys: LL a -> {zs:LL a | elems zs == elems xs + elems ys} @-}
helper :: LL a -> LL a -> LL a
helper acc Nil = acc
helper acc (Cons h l) = helper (Cons h acc) l
"""
sample_pair["test_prop"] = None
sample_pair["bad_func"] = "helper"

masked_code, masks, fixes = get_masked_lines(sample_pair["bad"], sample_pair["fix"])
fname ="List.lhs"
prefix = starcoder_prompt.format(file_name=fname)
good_pair = False
for m, f in zip(masks, fixes):
    if "{-@" in m and (sample_pair["bad_func"] in m.split() or sample_pair["bad_func"] in f.split()):
        # print(masked_code)
        # print(prefix)
        # sample_pair["fix"] = sample_pair["fix"].replace(" LH ", " LiquidHaskell ").replace("\n\n\n", "\n\n")
        # print(f">>> bad function := {sample_pair['bad_func']}")
        # print(m)
        # print("------------------------------")
        type_loc = m.find(sample_pair["bad_func"])
        # print(type_loc)
        new_mask = m[:]
        if type_loc > -1:
            end_loc = m[type_loc + len(sample_pair["bad_func"]):].find("@-}")
            new_mask = m[:type_loc + len(sample_pair["bad_func"])] + " :: <fimask> " + m[type_loc + len(sample_pair["bad_func"]) + end_loc:]
        if USE_BUGGY_LINES_AS_COMMENT:
            prefix_mask = "\n".join([f"-- {ml}" for ml in m.splitlines()])
            new_mask = f"{prefix_mask}\n{new_mask}"
        ground_truth = f[:]
        # print(new_mask)
        masked_code = masked_code.replace("<<mask>>", new_mask, 1)
        good_pair = True
    else:
        masked_code = masked_code.replace("<<mask>>", m, 1)

if good_pair:
    sample_pair["key"] = f"{fname}--{sample_pair['bad_func']}"
    sample_pair["prefix"] = prefix
    sample_pair["masked_code"] = prefix + masked_code
    if sample_pair["masked_code"].count("<fimask>") != 1:
        print(f"{sample_pair['masked_code'].count('<fimask>')} masks in final prompt")
    FIM_PREFIX = "<fim_prefix>"
    FIM_MIDDLE = "<fim_middle>"
    FIM_SUFFIX = "<fim_suffix>"
    split_code = sample_pair["masked_code"].split("<fimask>")
    sample_pair["bad"] = f"{FIM_PREFIX}{split_code[0]}{FIM_SUFFIX}{split_code[1]}{FIM_MIDDLE}"
    print(sample_pair["bad"])
    print("++++++++++++++++++++++++++++++")
    sample_pair["ground_truth_type"] = ground_truth
    json_dataset.append(sample_pair)
    ground_truth_types[sample_pair["key"]] = ground_truth.splitlines()

with open("liquid_haskell_test_single_file.jsonl", "w", encoding="utf-8") as out_file:
    for sample_dict in json_dataset:
        out_file.write(json.dumps(sample_dict) + '\n')

with open("liquid_haskell_test_single_file_labels.jsonl", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(ground_truth_types, indent=4) + '\n')
