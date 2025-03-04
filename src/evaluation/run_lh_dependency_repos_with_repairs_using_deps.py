import argparse
from collections import defaultdict, OrderedDict, Counter, deque
from os.path import join, exists
from torch import cuda
import gc
import json
import re
import subprocess as subp
from predict.get_starcoder_code_suggestions import StarCoderModel
from predict.get_codellama_code_suggestions import CodeLlamaModel
from evaluation.prog_diff import get_token_differences
from evaluation.extract_haskell_functions import extract_and_group_haskell_functions

LIQUID_PRAGMAS = set(["predicate", "embed", "liquid", "LIQUID", "data", "type", "measure", "inline", "assume", "ignore"])
GITHUB_REPO = None

def get_args():
    parser = argparse.ArgumentParser(description='run_dependency_lh_tests')
    parser.add_argument('--total_preds', default=10, type=int,
                        help='total type predictions to generate per function type (default: 10)')
    parser.add_argument('--max_preds', default=50, type=int,
                        help='maximum number of predictions to generate with the model in total (default: 50)')
    parser.add_argument('--llm', default="starcoderbase-3b",
                        help='llm to use for code generation {starcoderbase-1b, -3b, -15b, codellama-7b} (default: starcoderbase-3b)')
    parser.add_argument('--use_finetuned', action='store_true',
                        help='use finetuned LLM (default: False)')
    parser.add_argument('--use_qualifiers', action='store_true',
                        help='use qualifiers extracted from the predicted types (default: False)')
    parser.add_argument('--use_dependencies_in_prompt', action='store_true',
                        help='use function dependencies in the prompt for more context (default: False)')
    parser.add_argument('--update_cache', action='store_true',
                        help='update the prompt cache (default: False)')
    parser.add_argument('--use_cache', action='store_true',
                        help='use the prompt cache (default: False)')
    parser.add_argument('--create_cache_only', action='store_true',
                        help='only create the prompt cache and don\'t run any tests (default: False)')
    parser.add_argument('--cache_file', default="./benchmarks/hsalsa20_prompt_cache_starcoderbase_3b.json",
                        help='use the given file for prompt -> generation cache (default: "./benchmarks/hsalsa20_prompt_cache_starcoderbase_3b.json")')
    parser.add_argument('--print_logs', action="store_true", default=False,
                        help='print the log messages (default: False)')
    parser.add_argument('--exec_dir', default="../hsalsa20",
                        help='benchmark data directory for execution (default: ../hsalsa20)')
    parser.add_argument('--out_dir', default="./results",
                        help='output data directory (default: ./results)')
    _args = parser.parse_args()
    return _args


def clean_type(ltype):
    return ' '.join(ltype.split()).replace('{ ', '{').replace(' }', '}').strip()


def print_prog_liquid_types(prog):
    types_str = ""
    in_type = False
    for l in prog.split('\n'):
        if ("{-@" in l or "{--" in l) and not any(p in l for p in LIQUID_PRAGMAS):
            in_type = True
        if in_type:
            types_str += l + "\n"
        if "@-}" in l:
            in_type = False
    print("-" * 42 + "Types in program" + "-" * 42)
    print(types_str.strip())
    print("-" * 100)


def print_stack(stack):
    print("-- stack -------------------------------------")
    i = len(stack)
    for i, (fname, func) in enumerate(reversed(stack)):
        if i >= 5:
            i -= 1
            break
        print(f"{fname} --> {func}")
    print(f"...and {max(0, len(stack)-1-i)} more functions")
    print("----------------------------------------------")


class LiquidFile():
    def __init__(self, name, code):
        self.name = name
        self.code = code
        all_liquid_types = filter(lambda t: t.split()[1] not in LIQUID_PRAGMAS, re.findall(r"{-@[\s\S]*?@-}", code))
        self.liquid_types = {}
        self.liquid_funcs = set()
        self.ground_truths = {}
        self.using_ground_truth = {}
        self.current_types = OrderedDict()
        for t in all_liquid_types:
            func = t.split()[1].strip()
            ground_truth = re.findall(r"{-@\s*?" + func + r"\s*?::([\s\S]*?)@-}", code)[0].strip()
            self.liquid_types[func] = t
            self.liquid_funcs.add(func)
            self.ground_truths[func] = clean_type(ground_truth)
            self.using_ground_truth[func] = False
            self.current_types[func] = None
        file_prefix = name.replace(".hs", ".").replace("/", ".")
        exports = re.findall(r"module\s*?" + file_prefix[:-1] + r"\s*\(([\s\S]*?)\)\s*?where", code)
        if exports:
            exports = exports[0]
        else:
            exports = "<<None>>"
        self.orig_exports = exports
        self.exports = set(self.orig_exports.replace(",", " ").replace(file_prefix, "").strip().split())
        new_exports = self.orig_exports[:]
        new_exports = new_exports.rstrip().rstrip(",") + "\n"
        eparts = new_exports.split(",")
        if len(eparts) > 1 and eparts[-1].strip().startswith("--"):
            new_exports = ",".join(eparts[:-1]) + eparts[-1]
        if self.name == "Rowround.hs":
            new_exports += ", elts\n"
        for func in self.current_types:
            if self.name == "Rowround.hs" and func in ["algMapsCompute", "algMapsDisplay", "algMapsKeelung"]:
                continue
            if self.name == "Data/ByteString/Internal/Pure.hs" and func in ["go1", "go2", "go3", "go4", "go5"]:
                continue
            if func not in self.exports:
                new_exports += f", {func}\n"
        # FIXME: we probably need the next line, here or in some other form
        # self.code = self.code.replace(exports, new_exports)
        self.type_preds_cache = {func: [] for func in self.current_types} # To check locally,
        # how many predictions in this current state we've tested for this type (for backtracking)
        self.total_times_tested = {func: 0 for func in self.current_types} # To check globally,
        # how many predictions we've tested for this type (for stopping backtracking and using ground truth)
        self.num_of_llm_calls = {func: 0 for func in self.current_types}
        self.tested_types_num = {func: 0 for func in self.current_types}
        self.func_predicates = {func: [] for func in self.current_types}
        self.seen_predicates = set()

    def get_func_qualifiers(self, func):
        return self.func_predicates[func]

    def update_func_qualifiers(self, func):
        qualifier_pattern = r"\{([^\|\{\}]+?\s*?:\s*?[^\|\{\}]+?)\|([\s\S]+?)\}"
        idx = len(self.func_predicates[func])
        for t_pred in self.type_preds_cache[func]:
            clean_predicates = []
            for inputs, predicates in re.findall(qualifier_pattern, t_pred):
                i = re.sub(r":.+", ": a", inputs)
                if "&&" in predicates:
                    temp = predicates.strip().replace(' = ', ' == ')
                    split_predicates = temp.split("&&")
                    for p in split_predicates:
                        temp = p.strip()
                        if p.count('(') == p.count(')') and p.count("(") % 2 == 1:
                            temp = temp.strip('(').strip(')')
                        elif p.count('(') > p.count(')'):
                            temp = temp.strip('(')
                        elif p.count('(') < p.count(')'):
                            temp = temp.strip(')')
                        clean_predicates.append((i.strip(), p.strip()))
                else:
                    clean_predicates.append((i.strip(), predicates.strip().replace(' = ', ' == ')))

            clean_inputs = [i for i, _ in clean_predicates]
            for qual in t_pred.split('->'):
                if ':' in qual and "|" not in qual:
                    clean_qual = qual.strip().strip('(').strip(')').strip()
                    clean_qual = re.sub(r":.+", ": a", clean_qual)
                    clean_inputs.append(clean_qual)

            for i, p in clean_predicates:
                var_names = ["k", "l", "w", "z", "y", "x"]
                all_vars = [i]
                var_i = i.split(":")[0].strip()
                key = f"({i}): {p}".replace(var_i, var_names.pop())
                words = re.findall(r"[_a-z]\w*", p)
                for inp in clean_inputs:
                    var_inp = inp.split(":")[0].strip()
                    if inp != i and var_inp in words:
                        key.replace("):", inp + "):").replace(var_i, var_names.pop())
                        all_vars.append(inp)
                if key not in self.seen_predicates:
                    self.seen_predicates.add(key)
                    self.func_predicates[func].append(f"{{-@ qualif {func.capitalize()}_{idx}({', '.join(all_vars)}): {p} @-}}")
                    idx += 1

    def update_types_in_file(self, use_qualifiers):
        llm_prog = self.code
        qualifiers = []
        for func, ltype in self.current_types.items():
            pattern = re.escape(self.liquid_types[func]) # + r"\s*?\n"
            if ltype:
                # Replace type in file
                # NOTE: need to take care of possible '\f ->' in predicted types
                # '\\'s are evaluated to one '\' when subing, so we need to add another one
                if "\\" in ltype:
                    ltype = ltype.replace("\\", "\\\\")
                llm_prog = re.sub(pattern, f"{{-@ {func} :: {ltype} @-}}", llm_prog, 1)
                qualifiers.extend(self.func_predicates[func])
            else:
                # Ignore the function
                llm_prog = re.sub(pattern, f"{{-@ ignore {func} @-}}", llm_prog, 1)
        exports_idx = llm_prog.find(self.orig_exports)
        where_end_idx = llm_prog.find("where", exports_idx) + len("where")
        prefix = llm_prog[:where_end_idx]
        if use_qualifiers:
            llm_prog = llm_prog.replace(prefix, prefix + "\n\n" + "\n".join(qualifiers) + "\n")
        return llm_prog

    def set_func_type(self, func, ltype):
        if not self.using_ground_truth[func]:
            self.current_types[func] = ltype
        else:
            self.current_types[func] = self.ground_truths[func]

    def set_func_type_to_ground_truth(self, func, max_preds):
        self.current_types[func] = self.ground_truths[func]
        self.using_ground_truth[func] = True
        # self.type_preds_cache[func] = [self.ground_truths[func]]
        self.num_of_llm_calls[func] = max_preds
        self.tested_types_num[func] = 0

    def is_exported(self, func):
        return func in self.exports

    def make_prompt_for_func(self, func, code_snippets, llm):
        FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX = None, None, None
        if "starcoder" in llm:
            FIM_PREFIX = "<fim_prefix>"
            FIM_MIDDLE = "<fim_middle>"
            FIM_SUFFIX = "<fim_suffix>"
        elif "codellama" in llm:
            FIM_PREFIX = "<PRE> "
            FIM_MIDDLE = " <MID>"
            FIM_SUFFIX = " <SUF>"
        prefix = f"<filename>solutions/{self.name}\n"
        prefix += "-- Fill in the masked refinement type in the following LiquidHaskell program\n"

        llm_prog = self.code
        # FIXME: Hack to shorten prompt, because Crypt.hs is too long
        # TODO: Try to keep only dependencies or functions that have types
        if "codellama" in llm and self.name == "Quarterround.hs":
            llm_prog = "\n".join(llm_prog.split("\n")[:85] + llm_prog.split("\n")[130:])
        parts = llm_prog.split('-}')
        llm_prog = '-}'.join(parts[1:]).strip()
        if self.name == "Data/ByteString/Internal/Type.hs":
            llm_prog = "\n".join(llm_prog.split("\n")[:274])
        pattern = re.escape(self.liquid_types[func]) # + r"\s*?\n"
        # Mask the type that we want to predict for
        llm_prog = re.sub(pattern, f"{{-@ {func} :: <fimask> @-}}", llm_prog, 1)
        # Remove any types that we have not predicted for or used yet, and keep rest of the predictions
        for tfunc, ltype in self.current_types.items():
            if tfunc == func:
                continue
            pattern = re.escape(self.liquid_types[tfunc]) # + r"\s*?\n"
            if ltype:
                llm_prog = re.sub(pattern, f"{{-@ {func} :: {ltype} @-}}", llm_prog, 1)
            # FIXME: If don't have any types in the prompt in the first tries, the predictions are not good (few-shot learning)
            # TODO: Maybe add a few fixed examples, like the user provided them
            else:
                llm_prog = re.sub(pattern, "", llm_prog, 1)

        split_code = llm_prog.split("<fimask>")
        prompt = None
        if self.name == "Data/ByteString/Internal/Pure.hs":
            part_0 = "\n".join(split_code[0].split("\n")[-100:])
            part_1 = "\n".join(split_code[1].split("\n")[:100])
            if code_snippets:
                part_1 = "\n".join(split_code[1].split("\n")[:50])
            prompt = f"{FIM_PREFIX}{code_snippets}{prefix}{part_0}{FIM_SUFFIX}{part_1}{FIM_MIDDLE}"
        else:
            part_0 = split_code[0]
            part_1 = "\n".join(split_code[1].split("\n")[:100])
            if "codellama" in llm and code_snippets:
                part_0 = "\n".join(split_code[0].split("\n")[-75:])
            if code_snippets:
                part_1 = "\n".join(split_code[1].split("\n")[:50])
            prompt = f"{FIM_PREFIX}{code_snippets}{prefix}{part_0}{FIM_SUFFIX}{part_1}{FIM_MIDDLE}"
        return prompt

    def func_has_more_preds_avail(self, func):
        return self.tested_types_num[func] < len(self.type_preds_cache[func])

    def get_next_pred_for_func(self, func):
        next_pred_id = self.tested_types_num[func]
        # print(next_pred_id, len(self.type_preds_cache[func]))
        if next_pred_id < len(self.type_preds_cache[func]):
            type_pred = self.type_preds_cache[func][next_pred_id]
            # type_pred = list(reversed(self.type_preds_cache[func]))[next_pred_id]
            self.tested_types_num[func] = next_pred_id + 1
            return type_pred
        return None

    def print_type_state(self):
        for tfunc, ltype in self.current_types.items():
            preds = len(self.type_preds_cache[tfunc])
            total_tests = self.total_times_tested[tfunc]
            llm_calls = self.num_of_llm_calls[tfunc]
            curr_idx = self.tested_types_num[tfunc]
            if self.using_ground_truth[tfunc]:
                print(f">>> [[{tfunc:<23}: {curr_idx}/{preds} | {total_tests} tests | {llm_calls} llm calls | {ltype}]]")
            else:
                print(f">>> {tfunc:<25}: {curr_idx}/{preds} | {total_tests} tests | {llm_calls} llm calls | {ltype}")


class ProjectState():
    def __init__(self, exec_path, all_files, dependencies, max_preds, use_qualifiers):
        # Global state variables
        self.exec_path = exec_path
        self.src_path = exec_path
        if GITHUB_REPO == "hsalsa20":
            self.src_path = join(self.src_path, "src")
        self.max_preds = max_preds
        self.dependencies = dependencies
        self.files = OrderedDict()
        self.curr_file_code = {}
        self.upper_bounds = {}
        self.use_qualifiers = use_qualifiers
        # Clean repo
        print("Initializing and cleaning up repo...", flush=True)
        cmds = "source /home/gsakkas/.ghcup/env; " # for local Haskell installation
        cmds += "export PATH=$PATH:/home/gsakkas/usr/bin; " # for local Z3 installation
        cmds += f"cd {self.exec_path}; "
        if GITHUB_REPO == "hsalsa20":
            cmds += f"git restore src; "
        else:
            cmds += f"git restore Data; "
        cmds += f"stack build"
        subp.run(cmds, shell=True, check=False, capture_output=True)
        for filename in all_files:
            with open(join(self.src_path, filename), "r", encoding="utf-8") as prog:
                code = prog.read()
                self.files[filename] = LiquidFile(filename, code)
                self.curr_file_code[filename] = self.files[filename].code # To capture the updated exports
        self.seen_states = {}
        # Set function not present in the dependencies file to be the ground truth always
        # This way the user can provide some "easier" refinement types as examples
        if GITHUB_REPO == "bytestring":
            del self.files["Data/LiquidPtr.hs"]

        for key in list(self.dependencies.keys()):
            filename, func = key
            if filename not in self.files:
                del self.dependencies[key]
                continue
            file_obj = self.files[filename]
            if func not in file_obj.liquid_types:
                del self.dependencies[key]

        for key in list(self.dependencies.keys()):
            self.dependencies[key] = [(filename, func) for filename, func in self.dependencies[key] if filename in self.files and func in self.files[filename].liquid_types]

        # Delete functions that don't have refinement types
        for filename, file_obj in list(self.files.items()):
            if not file_obj.current_types:
                del self.files[filename]
                continue
            for func in file_obj.current_types:
                if (filename, func) not in self.dependencies:
                    file_obj.set_func_type_to_ground_truth(func, self.max_preds + 1)
        # Current state variables (so we don't have to pass them at every class method)
        self.func = None
        self.filename = None
        self.file_obj = None

    def print_all_type_states(self):
        for filename, file_obj in self.files.items():
            print(f"---- File: {filename}: -------------------------------------------------")
            file_obj.print_type_state()

    def update_current_state(self, filename, func):
        self.filename = filename
        self.func = func
        self.file_obj = self.files[filename]

    def get_all_file_func_pairs(self):
        all_funcs = []
        for filename, file_obj in self.files.items():
            for func in file_obj.current_types:
                if (filename, func) in self.dependencies:
                    all_funcs.append(((self.upper_bounds[(filename, func)], filename, func), filename, func))
        return [(fname, func) for _, fname, func in sorted(all_funcs, key=lambda x: x[0])]

    def set_file_func_type(self, ltype):
        self.file_obj.set_func_type(self.func, ltype)

    def set_file_func_to_ground(self):
        self.file_obj.set_func_type_to_ground_truth(self.func, self.max_preds + 1)
        # Clean up all dependencies prediction tries to retry them later if this fails
        func_deps = self.dependencies[(self.filename, self.func)]
        for dep_filename, dep_func in func_deps:
            # TODO: Check if dep_func uses ground truth type?
            self.files[dep_filename].tested_types_num[dep_func] = 0

    def clean_func(self):
        # self.file_obj.tested_types_num[self.func] = max(0, self.file_obj.tested_types_num[self.func] - 1)
        self.file_obj.tested_types_num[self.func] = 0
        # if not self.file_obj.using_ground_truth[self.func]:
        self.file_obj.current_types[self.func] = None

    def clean_func_dependants_and_add_to_stack(self, stack):
        # Clean up all functions depending on current state function
        # Make sure to not re-add funcions to function state stack
        # TODO: Maybe remove stack logic from here and keep it only in main loop?
        visited = set()
        key = (self.filename, self.func)
        queue = deque([key])
        while queue:
            key = queue.popleft()
            if key in visited:
                continue
            visited.add(key)
            # for dep_filename, dep_func in self.dependencies[key]:
            for next_key, deps in reversed(self.dependencies.items()):
                if key in deps and next_key not in visited and next_key not in queue:
                    print(f"Cleaning dependant key {next_key}...")
                    next_filename, next_func = next_key
                    next_file_obj = self.files[next_filename]
                    next_file_obj.tested_types_num[next_func] = 0
                    # next_file_obj.tested_types_num[next_func] = max(0, next_file_obj.tested_types_num[next_func] - 1)
                    next_file_obj.current_types[next_func] = None
                    queue.append(next_key)
                    if next_key not in stack:
                        print("And adding it to the stack...")
                        stack.append(next_key)

    def get_all_dependencies(self):
        key = (self.filename, self.func)
        deps = []
        for dep_filename, dep_func in self.dependencies[key]:
            deps.append((dep_filename, dep_func))
        return deps

    def get_least_used_dependency_not_in_stack(self, stack):
        root_key = (self.filename, self.func)
        queue = deque([root_key])
        visited = set()
        parents = {root_key: None}
        num_of_uses = float("inf")
        next_key = None
        while queue and num_of_uses == float("inf"):
            for _ in range(len(queue)):
                key = queue.popleft()
                if key in visited:
                    continue
                visited.add(key)
                for dep_filename, dep_func in self.dependencies[key]:
                    dep_file_obj = self.files[dep_filename]
                    candidate_key = (dep_filename, dep_func)
                    parents[candidate_key] = key
                    if candidate_key not in visited and candidate_key not in queue:
                        temp = dep_file_obj.total_times_tested[dep_func] / max(1, len(dep_file_obj.type_preds_cache[dep_func]))
                        queue.append(candidate_key)
                        if dep_file_obj.current_types[dep_func] and dep_file_obj.using_ground_truth[dep_func]:
                            # Skip since it has a type already, that is probably correct
                            print(dep_filename, dep_func, "is already ground truth")
                            continue
                        if candidate_key in stack:
                            print(dep_filename, dep_func, "in stack")
                            continue
                        if temp < num_of_uses:
                            num_of_uses = temp
                            next_key = candidate_key
                            print(f"candidate key ({next_key}) with {num_of_uses} uses so far")
        res = []
        while next_key and parents[next_key]:
            res.append(next_key)
            next_key = parents[next_key]
        return res

    def get_type_state_key(self):
        all_states = []
        for filename, file_obj in self.files.items():
            for func, ltype in file_obj.current_types.items():
                if ltype:
                    all_states.append((f"{filename}.{func}", ltype))
        type_state = ""
        for file_func, ltype in sorted(all_states):
            type_state += f"{file_func} :: {ltype}<->"
        type_state = type_state[:-3]
        return type_state

    def get_all_dependencies_code_snippets(self):
        dep_files = defaultdict(list)
        for dep_filename, dep_func in self.get_all_dependencies():
            new_code = self.files[dep_filename].update_types_in_file(self.use_qualifiers)
            file_snippets = extract_and_group_haskell_functions(new_code)
            if dep_func in file_snippets:
                dep_files[dep_filename].append(file_snippets[dep_func])
        all_code_snippets = ""
        for dep_filename, all_dep_functions_code in dep_files.items():
            all_code_snippets += f"<filename>solutions/{dep_filename}\n"
            for code_snippet in all_dep_functions_code:
                all_code_snippets += code_snippet + "\n\n"

        print(all_code_snippets)
        return all_code_snippets

    def verify_project(self):
        self.file_obj.total_times_tested[self.func] += 1
        # Write to files only here to avoid unnecessary writes
        for filename, file_obj in self.files.items():
            new_code = file_obj.update_types_in_file(self.use_qualifiers)
            # print(f"------ File: {filename}:")
            # print_prog_liquid_types(new_code)
            if self.curr_file_code[filename] != new_code:
                self.curr_file_code[filename] = new_code
                with open(join(self.src_path, filename), "w", encoding="utf-8") as prog:
                    prog.write(new_code)
        str_state = self.get_type_state_key()
        if str_state in self.seen_states:
            print("Tested before.....")
            if not self.seen_states[str_state]:
                self.file_obj.total_times_tested[self.func] += len(self.file_obj.type_preds_cache[self.func]) - 1
            return self.seen_states[str_state]
        self.seen_states[str_state] = False
        # Just a random check, cause LH crashes for too long types
        if len(self.file_obj.current_types[self.func]) > len(self.file_obj.ground_truths[self.func]) + 64:
            return self.seen_states[str_state]
        cmds = "source /home/gsakkas/.ghcup/env; " # for local Haskell installation
        cmds += "export PATH=$PATH:/home/gsakkas/usr/bin; " # for local Z3 installation
        cmds += f"cd {self.exec_path}; "
        cmds += f"stack build"
        # cmds += f"git restore src"
        test_output = subp.run(cmds, shell=True, check=False, capture_output=True)
        result = test_output.stderr.decode('utf-8').strip()
        while "Cannot parse specification:" in result and "{-@ qualif" in result:
            print("Problem with parsing qualifiers! Deleting:")
            qualif_error_locations = [rline.split(".hs")[0] + ".hs" for rline in result.split("\n") if "error:" in rline]
            qualifs_with_errors = ["{-@ qualif" + rline.split("{-@ qualif")[1].split("@-}")[0] + "@-}" for rline in result.split("\n") if "{-@ qualif" in rline]
            qualif_errors = list(zip(qualif_error_locations, qualifs_with_errors))
            print("\n".join([f"{eloc}:\n{q}" for eloc, q in qualif_errors]))
            for filename, file_obj in self.files.items():
                new_code = self.curr_file_code[filename][:]
                for eloc, q in qualif_errors:
                    if filename in eloc:
                        new_code = new_code.replace(q + "\n", "")
                        all_predicates = self.files[filename].func_predicates
                        for func in all_predicates:
                            if func.capitalize() in q and q in all_predicates[func]:
                                all_predicates[func].remove(q)

                if self.curr_file_code[filename] != new_code:
                    self.curr_file_code[filename] = new_code
                    with open(join(self.src_path, filename), "w", encoding="utf-8") as prog:
                        prog.write(new_code)
            print("Deleted problematic qualifiers; Retrying verification...")
            test_output = subp.run(cmds, shell=True, check=False, capture_output=True)
            result = test_output.stderr.decode('utf-8').strip()
        # print("<<" * 42)
        # if GITHUB_REPO == "hsalsa20":
        #     print("/home/gsakkas/hsalsa20".join(result.split("/home/gsakkas/hsalsa20")[1:]))
        # else:
        #     print("/home/gsakkas/bytestring_lh-llm".join(result.split("/home/gsakkas/bytestring_lh-llm")[1:]))
        # print(">>" * 42)
        if GITHUB_REPO == "hsalsa20" and \
            result != "" and "UNSAFE" not in result and "Error:" not in result and "SAFE" in result:
            self.seen_states[str_state] = True
        elif GITHUB_REPO == "bytestring" and \
            result != "" and ("UNSAFE" not in result and "[17 of 34] Compiling Data.ByteString.Builder.RealFloat.Internal" in result):
            self.seen_states[str_state] = True
        # if not self.seen_states[str_state]:
        #     self.file_obj.total_times_tested[self.func] += len(self.file_obj.type_preds_cache[self.func]) - 1
        return self.seen_states[str_state]

    def is_using_ground_truth(self):
        if self.file_obj.using_ground_truth[self.func]:
            # Reinstate ground truth type, so we have the correct one when we verify
            # NOTE: We may have deleted it with clear_func before, if ground truth failed
            self.file_obj.set_func_type_to_ground_truth(self.func, self.max_preds + 1)
            return True
        return False

    def llm_calls_less_than_max(self, filename, func):
        return self.files[filename].num_of_llm_calls[func] < self.max_preds

    def state_llm_calls_less_than_max(self):
        return self.file_obj.num_of_llm_calls[self.func] < self.max_preds

    def deps_have_been_proven(self):
        key = (self.filename, self.func)
        for dep_filename, dep_func in self.dependencies[key]:
            dep_file_obj = self.files[dep_filename]
            # if not dep_file_obj.current_types[dep_func] and not dep_file_obj.using_ground_truth[dep_func]:
            if not dep_file_obj.current_types[dep_func]:
                return False
        return True

    def has_more_preds_avail(self):
        return self.file_obj.func_has_more_preds_avail(self.func)

    def get_next_pred(self):
        return self.file_obj.get_next_pred_for_func(self.func)

    def get_times_tested_per_pred(self):
        # Metric to avoid running too many times a type with too few predictions
        # return ceil(self.file_obj.total_times_tested[self.func] / sqrt(len(self.file_obj.type_preds_cache[self.func])))
        return self.file_obj.total_times_tested[self.func] // max(1, len(self.file_obj.type_preds_cache[self.func]))

    def get_functions_with_no_dependants(self):
        all_keys = self.get_all_file_func_pairs()
        for deps in self.dependencies.values():
            for dep_key in deps:
                if dep_key in all_keys:
                    all_keys.remove(dep_key)
        return all_keys

    def set_test_run_limits(self):
        self.upper_bounds = {}
        queue = deque()
        for key, deps in self.dependencies.items():
            if len(deps) == 0:
                queue.append(key)
                self.upper_bounds[key] = 10
        visited = set()
        while queue:
            key = queue.popleft()
            if key in visited:
                continue
            visited.add(key)
            for next_key, deps in self.dependencies.items():
                if key in deps and next_key not in visited and next_key not in queue:
                    queue.append(next_key)
                    self.upper_bounds[next_key] = max(2, self.upper_bounds[key] - 1)
        return self.upper_bounds


def get_type_predictions(prompt, filename, func_name, ground_truth, llm, args):
    print("-" * 42)
    print(f"New type predictions for ({filename}, {func_name})")
    print("-> LLM generation...", flush=True)
    # NOTE: The deeper we are in the loop, get more predictions
    # in order to avoid backtracking and potentially removing a correct earlier type
    # Potentially, can be done differently, i.e. generate more when failing
    prog_preds = llm.get_code_suggestions(prompt, min(args.max_preds, args.total_preds))
    # NOTE: For HSalsa20, UInt 32 not in Liquid Haskell
    prog_preds = [clean_type(pred).replace("UInt 32", "_").replace("UInt32", "_") for pred in prog_preds]
    prog_preds = list(filter(lambda p: p.count('->') == ground_truth.count('->'), prog_preds))
    freq_map = Counter(filter(lambda p: func_name not in p, prog_preds))
    prog_preds = [pred for pred, _ in freq_map.most_common(args.total_preds)]
    print(f"-> {len(prog_preds[:10])} unique predicted types", flush=True)
    # Try to clean up memoru to avoid CUDA OOM after only a few iterations
    cuda.empty_cache()
    gc.collect()
    return prog_preds[:10]


def is_ground_truth_in_top_n_preds(func, ground_truth, type_preds, n=5):
    top_preds = type_preds[:n]
    if ground_truth in top_preds:
        print(f"These '{func}' types are the same: {ground_truth} {ground_truth}")
        return True
    for type_pred in top_preds:
        orig_parts: list[str] = []
        pred_parts: list[str] = []
        _, orig_parts, pred_parts = get_token_differences(ground_truth, type_pred)
        aliases = {}
        are_different = False
        for opart, ppart in zip(orig_parts, pred_parts):
            if "{" in opart or "}" in opart or ":" in opart or "|" in opart:
                are_different = True
                break
            if "{" in ppart or "}" in ppart or ":" in ppart or "|" in ppart:
                are_different = True
                break
            if opart.isnumeric() or ppart.isnumeric():
                are_different = True
                break
            if opart == "_" and ppart != "_":
                are_different = True
                break
            if opart != "_" and ppart != "_" and opart not in aliases:
                aliases[opart] = ppart
            elif opart in aliases and aliases[opart] != ppart:
                are_different = True
                break
        if not are_different:
            print(f"These '{func}' types are similar: {ground_truth} {type_pred}")
            return True
    return False


def run_tests(args):
    dependencies = {}
    prompt_cache = {}
    dep_file_dir = None
    if GITHUB_REPO == "hsalsa20":
        dep_file_dir = "./benchmarks/hsalsa20_dependencies_with_initialization_and_indirect_deps.json"
    else:
        dep_file_dir = "./benchmarks/bytestring_dependencies.json"
    with open(dep_file_dir, "r", encoding="utf-8") as dep_file:
        dependencies = json.loads(dep_file.read())
    if exists(args.cache_file):
        with open(args.cache_file, "r", encoding="utf-8") as cache_file:
            prompt_cache = json.loads(cache_file.read())
    fixed_progs = 0
    total_num_of_funcs_per_file = defaultdict(int)
    curr_num_of_funcs_tested = {}
    total_num_of_correct_funcs = {}
    runs_upper_bound = {}
    # all_files = [filename for filename in sorted(listdir(path)) if not filename.endswith(".hs") and not filename.endswith(".lhs")]
    all_files = []
    for key in dependencies:
        _fname = key.split("--")[0].strip()
        if _fname not in all_files:
            all_files.append(_fname)

    new_dependencies = {}
    for key in dependencies:
        filename = key.split("--")[0].strip()
        func = key.split("--")[1].strip()
        if (filename, func) not in new_dependencies:
            new_dependencies[(filename, func)] = [tuple(l) for l in dependencies[key]] #Because we had lists of 2 elements in dependnecies file
    dependencies = new_dependencies
    # NOTE: Clean-up in case we are missing some file-function pairs?
    for filename, func in dependencies:
        deps = dependencies[(filename, func)]
        dependencies[(filename, func)] = [t for t in deps if t in deps]

    # Load (finetuned) code LLM
    if "starcoder" in args.llm:
        code_llm = StarCoderModel(args.use_finetuned)
    elif "codellama" in args.llm:
        code_llm = CodeLlamaModel(args.use_finetuned)

    # Initialize Liquid Haskell project state with all the files and function dependencies
    project_state = ProjectState(args.exec_dir, all_files, dependencies, args.max_preds, args.use_qualifiers)

    for filename, func in project_state.dependencies:
        total_num_of_funcs_per_file[filename] += 1
        total_num_of_correct_funcs[filename] = 0
        curr_num_of_funcs_tested[filename] = 0

    total_num_of_progs = len(project_state.files)
    num_of_iterations = 0
    # num_of_skipped = 0
    runs_upper_bound = project_state.set_test_run_limits()
    for i in runs_upper_bound.items():
        print(i)
    total_llm_calls = 0
    func_stack = list(reversed(project_state.get_all_file_func_pairs()))
    root_funcs = list(reversed(project_state.get_functions_with_no_dependants()))
    for key in root_funcs:
        func_stack.remove(key)
    func_stack.extend(root_funcs)
    MAX_ITERATIONS = len(func_stack) * (args.max_preds * 3)
    func_stack = deque(func_stack)
    # print(runs_upper_bound, flush=True)
    while func_stack:
        filename, func = key = func_stack.pop()
        project_state.update_current_state(filename, func)
        file_obj = project_state.files[filename]
        assert isinstance(file_obj, LiquidFile) # NOTE: Just for autocomplete puproses later on
        # If we have generated too many types for all functions
        if all(not project_state.llm_calls_less_than_max(fname, fnc) for fname, fnc in project_state.get_all_file_func_pairs()):
            print(f"Reached limit of predictions ({args.max_preds}) for all functions; Exiting...", flush=True)
            break
        # If we had too many iteratin with the whole loop
        if num_of_iterations >= MAX_ITERATIONS:
            print(f"Too many iterations {num_of_iterations}; Exiting...", flush=True)
            break

        print("=" * 42)
        print(f"Solving {filename} ({func})...", flush=True)
        if not project_state.deps_have_been_proven():
            # Not all dependencies have a type, thus no need to check this key yet
            print(f"Not all dependencies are done yet... Pushing them up the stack...")
            func_stack.append(key)
            # Add least tested dependency to stack
            all_deps = project_state.get_all_dependencies()
            for dep in all_deps:
                if dep in func_stack:
                    print(f"{dep} already in stack, pushing up...", flush=True)
                    func_stack.remove(dep)
                    func_stack.append(dep)
                elif not project_state.files[dep[0]].current_types[dep[1]]:
                    print(f"Adding {dep} in stack...", flush=True)
                    func_stack.append(dep)
            continue
        print_stack(func_stack)
        num_of_iterations += 1
        solved = False
        if not project_state.is_using_ground_truth() and project_state.state_llm_calls_less_than_max() and \
                (file_obj.tested_types_num[func] == 0 or not project_state.has_more_preds_avail()):
            dependencies_code_snippets = ""
            if args.use_dependencies_in_prompt:
                dependencies_code_snippets = project_state.get_all_dependencies_code_snippets()
            prompt = file_obj.make_prompt_for_func(func, dependencies_code_snippets, args.llm)
            # Prompt key includes all predicted types so far, the current file and function and how many times we called with this prompt
            prompt_key = filename + "--" + func + "<-->" + project_state.get_type_state_key() + "<-->" + str(file_obj.num_of_llm_calls[func] // args.total_preds)
            if prompt_key not in prompt_cache:
                prompt_cache[prompt_key] = list(enumerate(get_type_predictions(prompt, filename, func, file_obj.ground_truths[func], code_llm, args)))
                with open(args.cache_file, "w", encoding="utf-8") as cache_file:
                    cache_file.write(json.dumps(prompt_cache, indent=4))
            type_preds = list(map(lambda p: p[1], sorted(prompt_cache[prompt_key], key=lambda p: p[0])))
            print(f"Got from cache or LLM {len(type_preds)} type predictions...")
            if file_obj.is_exported(func):
                type_preds = [tpred for tpred in type_preds if tpred.split("->")[-1].strip() not in ["_", "[_]"]]
                print(f"Kept only {len(type_preds)} type predictions for exported function...")
            elif args.use_qualifiers:
                parts = len(file_obj.ground_truths[func].split("->"))
                naive_type = " -> ".join(["_"] * parts)
                print(f"Added {naive_type} to type predictions for local function...")
                if naive_type in type_preds:
                    type_preds.remove(naive_type)
                type_preds.append(naive_type)
            total_llm_calls += 1
            num_of_preds = file_obj.num_of_llm_calls[func] + args.total_preds
            prev_len = len(file_obj.type_preds_cache[func])
            all_preds = file_obj.type_preds_cache[func]
            for pred in type_preds:
                if pred not in all_preds:
                    all_preds.append(pred)
            # If the LLM can't generate any new types, then try the ground truth type or go back
            # NOTE: Or if LLM generates too many different types (heuristic), probably they are not that good
            if len(all_preds) == prev_len:
                print("No new predictions...", flush=True)
            # elif (len(all_preds) > 20 or len(all_preds) == prev_len or len(all_preds) == 0) and not project_state.is_using_ground_truth():
            if not project_state.is_using_ground_truth():
                file_obj.type_preds_cache[func] = all_preds
                file_obj.num_of_llm_calls[func] = num_of_preds
            if len(all_preds) == 0 and not project_state.is_using_ground_truth() and not project_state.state_llm_calls_less_than_max():
                print(f"Testing the ground truth type, since we got no predictions...", flush=True)
                # print(f"Testing the ground truth type, since we got too many unique or no predictions ({len(all_preds)})...", flush=True)
                project_state.set_file_func_to_ground()

            if args.use_qualifiers:
                file_obj.update_func_qualifiers(func)

        for q in file_obj.get_func_qualifiers(func):
            print(q)

        if project_state.is_using_ground_truth():
            print("-" * 42)
            print(f"Testing {{-@ {func} :: {file_obj.ground_truths[func]} @-}}...", flush=True)
            solved = project_state.verify_project()
            if solved:
                print("...SAFE", flush=True)
            else:
                print("...UNSAFE", flush=True)
            project_state.print_all_type_states()
        else:
            type_prediction = project_state.get_next_pred()
            cnt = 0
            while type_prediction and cnt < args.max_preds:
                cnt += 1
                print("-" * 42)
                print(f"Testing {{-@ {func} :: {type_prediction} @-}}...", flush=True)
                if project_state.get_times_tested_per_pred() >= runs_upper_bound[key]:
                    print("Too many failures for this type; Testing the ground truth type...")
                    project_state.set_file_func_to_ground()
                    type_prediction = file_obj.ground_truths[func]
                    print(f"Testing {{-@ {func} :: {type_prediction} @-}}...", flush=True)
                else:
                    project_state.set_file_func_type(type_prediction)
                solved = project_state.verify_project()
                if solved:
                    print("...SAFE", flush=True)
                    break
                else:
                    print("...UNSAFE", flush=True)
                    type_prediction = project_state.get_next_pred()
                if project_state.is_using_ground_truth():
                    break
            project_state.print_all_type_states()
            if cnt == args.max_preds:
                print("Something went wrong with getting next type prediction!")

        print("-" * 42)
        print("-" * 42)
        if solved:
            print(f"{key} --> SAFE", flush=True)
            continue

        print(f"{key} --> UNSAFE", flush=True)
        if dependencies[key]:
            project_state.clean_func_dependants_and_add_to_stack(func_stack)
            # if not project_state.is_using_ground_truth():
            project_state.clean_func()
            # Add failed function to retry later
            func_stack.append(key)
            # Add least tested dependency to stack
            next_keys = project_state.get_least_used_dependency_not_in_stack(func_stack)
            if next_keys:
                next_obj = None
                next_key = None
                next_file, next_func = None, None
                for next_key in reversed(next_keys):
                    next_file, next_func = next_key
                    if next_key in func_stack:
                        print(f"{next_key} already in stack, pushing up...", flush=True)
                        func_stack.remove(next_key)
                    else:
                        print(f"Adding {next_key} in stack...", flush=True)
                    func_stack.append(next_key)
                    next_obj = project_state.files[next_file]
                    if not project_state.is_using_ground_truth():
                        next_obj.tested_types_num[next_func] = max(0, next_obj.tested_types_num[next_func] - 1)
                    else:
                        next_obj.tested_types_num[next_func] = 0
                    next_obj.current_types[next_func] = None
                    next_obj.total_times_tested[next_func] = max(0, next_obj.total_times_tested[next_func] - 1)
                if not project_state.is_using_ground_truth():
                    next_obj.tested_types_num[next_func] += 1
                next_obj.total_times_tested[next_func] += 1
                print(f"Backtracking to dependent function = {next_key}...", flush=True)
            else:
                print(f"Not all dependencies are correct yet... Pushing them up the stack...")
                all_deps = project_state.get_all_dependencies()
                for dep in all_deps:
                    if dep in func_stack:
                        print(f"{dep} already in stack, pushing up...", flush=True)
                        func_stack.remove(dep)
                        func_stack.append(dep)
                    elif not project_state.files[dep[0]].current_types[dep[1]]:
                        print(f"Adding {dep} in stack...", flush=True)
                        func_stack.append(dep)
        # elif not project_state.is_using_ground_truth() and project_state.get_times_tested_per_pred() < runs_upper_bound[key]:
        #     print(f"Trying again function {key}...", flush=True)
        #     project_state.clean_func_dependants_and_add_to_stack(func_stack)
        #     project_state.clean_func()
        #     func_stack.append(key)
        else:
            print(f"Trying again function {key}...", flush=True)
            project_state.clean_func_dependants_and_add_to_stack(func_stack)
            project_state.clean_func()
            # Add failed function to retry later
            func_stack.append(key)
            # func_stack.appendleft(key)

    total_ground_truths = defaultdict(int)
    total_num_of_ground_truths_used = 0
    total_num_of_refinemet_types = 0
    user_confirmed_types = 0
    for filename, file_obj in project_state.files.items():
        given_by_user = 0
        assert isinstance(file_obj, LiquidFile)
        for func in file_obj.current_types:
            if (filename, func) not in project_state.dependencies:
                given_by_user += 1
            elif file_obj.using_ground_truth[func]: # and is_ground_truth_in_top_n_preds(func, file_obj.ground_truths[func], file_obj.type_preds_cache[func]):
                # print(is_ground_truth_in_top_n_preds(func, file_obj.ground_truths[func], file_obj.type_preds_cache[func]))
                # Temporarily unset ground truth, so we can use predictions again
                file_obj.using_ground_truth[func] = False
                project_state.update_current_state(filename, func)
                for idx, type_prediction in enumerate(file_obj.type_preds_cache[func]):
                    project_state.set_file_func_type(type_prediction)
                    solved = project_state.verify_project()
                    if solved:
                        print(f"These '{func}' types are similar:")
                        print(f"ground truth: {file_obj.ground_truths[func]}")
                        print(f"prediction {idx+1} : {type_prediction}")
                        total_ground_truths[filename] -= 1
                        user_confirmed_types += 1
                        break
                project_state.set_file_func_to_ground()
        total_ground_truths[filename] += sum(file_obj.using_ground_truth.values()) - given_by_user
        total_num_of_ground_truths_used += total_ground_truths[filename]
        total_num_of_refinemet_types += total_num_of_funcs_per_file[filename]
        total_num_of_correct_funcs[filename] = sum([1 if f else 0 for f in file_obj.current_types.values()]) - given_by_user

    print("=" * 42)
    print("=" * 42)
    for filename in sorted(total_num_of_correct_funcs.keys()):
        if total_num_of_funcs_per_file[filename] > 0:
            print(f">>> File {filename} ({total_num_of_funcs_per_file[filename]} refinement types)")
            if total_num_of_correct_funcs[filename] == total_num_of_funcs_per_file[filename]:
                fixed_progs += 1
            else:
                print(f"File was not fully verified!!!")
            print(f"{total_num_of_correct_funcs[filename]} / {total_num_of_funcs_per_file[filename]} types predicted correctly")
            print(f"{total_ground_truths[filename]} / {total_num_of_funcs_per_file[filename]} ground truth types used")
            print(f"{(total_num_of_correct_funcs[filename] - total_ground_truths[filename]) * 100 / total_num_of_funcs_per_file[filename]:.2f}% prediction accuracy")
            print("-" * 42)
    print("=" * 42)
    print("=" * 42)
    print(f"{fixed_progs} / {total_num_of_progs} programs fully annotated correctly with LH types")
    print(f"{total_num_of_ground_truths_used} / {total_num_of_refinemet_types} used the ground truth user type")
    print(f"{user_confirmed_types} user-confirmed predicted types")
    print(f"{num_of_iterations} loop iterations (i.e. total number of types checked)")
    print(f"{total_llm_calls} llm calls of {args.total_preds} type predictions")


if __name__ == "__main__":
    cmd_args = get_args()
    if "hsalsa20" in cmd_args.exec_dir:
        GITHUB_REPO = "hsalsa20"
    else:
        GITHUB_REPO = "bytestring"
    run_tests(cmd_args)