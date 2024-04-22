import argparse
from collections import defaultdict, OrderedDict, Counter, deque
from os.path import join, exists
from torch import cuda
import gc
import json
from random import shuffle
import re
import subprocess as subp
from predict.get_starcoder_code_suggestions import StarCoderModel

LIQUID_PRAGMAS = set(["LIQUID", "data", "type", "measure", "inline", "assume", "ignore"])


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
    i = len(stack)-1
    for i, (fname, func) in enumerate(reversed(stack)):
        if i >= 5:
            break
        print(f"{fname} --> {func}")
    print(f"...and {len(stack)-1-i} more functions")
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
        exports = code.split("    )\nwhere")[0].split('(')[-1].rstrip().rstrip(",")
        new_exports = exports[:]
        if self.name == "Rowround.hs":
            new_exports += ", elts"
        for func in self.current_types:
            if self.name == "Rowround.hs" and func in ["algMapsCompute", "algMapsDisplay", "algMapsKeelung"]:
                continue
            if func not in exports:
                new_exports += f", {func}"
        self.code = self.code.replace(exports, new_exports)
        self.type_preds_cache = {func: [] for func in self.current_types} # To check locally,
        # how many predictions in this current state we've tested for this type (for backtracking)
        self.total_times_tested = {func: 0 for func in self.current_types} # To check globally,
        # how many predictions we've tested for this type (for stopping backtracking and using ground truth)
        self.num_of_llm_calls = {func: 0 for func in self.current_types}
        self.tested_types_num = {func: 0 for func in self.current_types}

    def update_types_in_file(self):
        llm_prog = self.code
        for func, ltype in self.current_types.items():
            pattern = re.escape(self.liquid_types[func]) + r"\s*?\n"
            if ltype:
                # Replace type in file
                # NOTE: need to take care of possible '\f ->' in predicted types
                # '\\'s are evaluated to one '\' when subing, so we need to add another one
                if "\\" in ltype:
                    ltype = ltype.replace("\\", "\\\\")
                llm_prog = re.sub(pattern, f"{{-@ {func} :: {ltype} @-}}\n", llm_prog, 1)
            else:
                # Ignore the function
                llm_prog = re.sub(pattern, f"{{-@ ignore {func} @-}}\n", llm_prog, 1)
        return llm_prog

    def set_func_type(self, func, ltype):
        if not self.using_ground_truth[func]:
            self.current_types[func] = ltype
        else:
            self.current_types[func] = self.ground_truths[func]

    def set_func_type_to_ground_truth(self, func, max_preds):
        self.current_types[func] = self.ground_truths[func]
        self.using_ground_truth[func] = True
        self.type_preds_cache[func] = [self.ground_truths[func]]
        self.num_of_llm_calls[func] = max_preds
        self.tested_types_num[func] = 0

    def make_prompt_for_func(self, func):
        FIM_PREFIX = "<fim_prefix>"
        FIM_MIDDLE = "<fim_middle>"
        FIM_SUFFIX = "<fim_suffix>"

        prefix = f"<filename>solutions/{self.name}\n-- Fill in the masked refinement type in the following LiquidHaskell program\n"
        llm_prog = self.code
        # FIXME: Hack to shorten prompt, because Crypt.hs is too long
        # TODO: Try to keep only dependencies or functions that have types
        parts = llm_prog.split('-}')
        llm_prog = '-}'.join(parts[1:]).strip()
        pattern = re.escape(self.liquid_types[func]) + r"\s*?\n"
        # Mask the type that we want to predict for
        llm_prog = re.sub(pattern, f"{{-@ {func} :: <fimask> @-}}\n", llm_prog, 1)
        # Remove any types that we have not predicted for or used yet, and keep rest of the predictions
        for tfunc, ltype in self.current_types.items():
            if tfunc == func:
                continue
            pattern = re.escape(self.liquid_types[tfunc]) + r"\s*?\n"
            if ltype:
                llm_prog = re.sub(pattern, f"{{-@ {func} :: {ltype} @-}}\n", llm_prog, 1)
            # FIXME: If don't have any types in the prompt in the first tries, the predictions are not good (few-shot learning)
            # TODO: Maybe add a few fixed examples, like the user provided them
            else:
                llm_prog = re.sub(pattern, "", llm_prog, 1)

        split_code = llm_prog.split("<fimask>")
        prompt = f"{FIM_PREFIX}{prefix}{split_code[0]}{FIM_SUFFIX}{split_code[1]}{FIM_MIDDLE}"
        return prompt

    def func_has_more_preds_avail(self, func):
        return self.tested_types_num[func] < len(self.type_preds_cache[func])

    def get_next_pred_for_func(self, func):
        next_pred_id = self.tested_types_num[func]
        # print(next_pred_id, len(self.type_preds_cache[func]))
        if next_pred_id < len(self.type_preds_cache[func]):
            type_pred = self.type_preds_cache[func][next_pred_id]
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
                print(f">>> [[{tfunc}: {curr_idx}/{preds} | {total_tests} tests | {llm_calls} llm calls | {ltype}]]")
            else:
                print(f">>> {tfunc}: {curr_idx}/{preds} | {total_tests} tests | {llm_calls} llm calls | {ltype}")


class ProjectState():
    def __init__(self, path, exec_path, all_files, dependencies, max_preds):
        # Global state variables
        self.path = path
        self.exec_path = exec_path
        self.max_preds = max_preds
        self.dependencies = dependencies
        self.files = OrderedDict()
        self.curr_file_code = {}
        # Clean repo
        print("Initializing and cleaning up repo...", flush=True)
        cmds = "source /home/gsakkas/.ghcup/env; " # for local Haskell installation
        cmds += "export PATH=$PATH:/home/gsakkas/usr/bin; " # for local Z3 installation
        cmds += f"cd {self.exec_path}; "
        cmds += f"git restore src; "
        cmds += f"stack build"
        subp.run(cmds, shell=True, check=False, capture_output=True)
        for filename in all_files:
            with open(join(path, filename), "r", encoding="utf-8") as prog:
                code = prog.read()
                self.files[filename] = LiquidFile(filename, code)
                self.curr_file_code[filename] = self.files[filename].code # To capture the updated exports
        self.seen_states = {}
        # Set function not present in the dependencies file to be the ground truth always
        # This way the user can provide some "easier" refinement types as examples
        for filename, file_obj in self.files.items():
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
                    all_funcs.append(((len(self.dependencies[(filename, func)]), filename, func), filename, func))
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
        self.file_obj.tested_types_num[self.func] = 0
        # if not self.file_obj.using_ground_truth[self.func]:
        self.file_obj.current_types[self.func] = None

    def clean_func_dependants_and_add_to_stack(self, stack):
        # Clean up all functions depending on current state function
        # Make sure to not re-add funcions to function state stack
        # TODO: Maybe remove stack logic from here and keep it only in main loop?
        key = (self.filename, self.func)
        for next_key, deps in reversed(self.dependencies.items()):
            next_filename, next_func = next_key
            next_file_obj = self.files[next_filename]
            if key in deps:
                print(f"Cleaning dependant key {next_key}...")
                next_file_obj.tested_types_num[next_func] = 0
                next_file_obj.current_types[next_func] = None
                if next_key not in stack:
                    print("And adding it to the stack...")
                    stack.appendleft(next_key)

    def get_least_tested_dependency_not_in_stack(self, stack):
        key = (self.filename, self.func)
        all_deps_are_set = self.deps_have_been_proven()
        next_key = None
        # next_key = self.dependencies[key][0]
        # next_filename, next_func = next_key
        # next_file_obj = self.files[next_filename]
        untested_types = 100000 # len(next_file_obj.type_preds_cache[next_func]) - next_file_obj.tested_types_num[next_func]
        for dep_filename, dep_func in self.dependencies[key]:
            dep_file_obj = self.files[dep_filename]
            if dep_file_obj.current_types[dep_func] and dep_file_obj.using_ground_truth[dep_func]:
                # Skip since it has a type already, that is probably correct
                print(dep_filename, dep_func, "is already ground truth")
                continue
            if not all_deps_are_set and dep_file_obj.current_types[dep_func]:
                print(dep_filename, dep_func, "has a type but others are unassigned")
                continue
            if (dep_filename, dep_func) in stack:
                print(dep_filename, dep_func, "in stack")
                continue
            # temp = len(dep_file_obj.type_preds_cache[dep_func]) - dep_file_obj.tested_types_num[dep_func]
            temp = dep_file_obj.total_times_tested[dep_func]
            if dep_file_obj.total_times_tested[dep_func] < untested_types:
                untested_types = temp
                next_key = (dep_filename, dep_func)
        return next_key

    def get_type_state_key(self):
        type_state = ""
        for filename, file_obj in self.files.items():
            for func, ltype in file_obj.current_types.items():
                if ltype:
                    type_state += f"{filename}.{func} :: {ltype}<->"
        type_state = type_state[:-3]
        return type_state

    def verify_project(self):
        self.file_obj.total_times_tested[self.func] += 1
        key = (self.filename, self.func)
        for dep_filename, dep_func in self.dependencies[key]:
            dep_file_obj = self.files[dep_filename]
            dep_file_obj.total_times_tested[dep_func] += 1
        # Write to files only here to avoid unnecessary writes
        for filename, file_obj in self.files.items():
            new_code = file_obj.update_types_in_file()
            # print(f"------ File: {filename}:")
            # print_prog_liquid_types(new_code)
            if self.curr_file_code[filename] != new_code:
                self.curr_file_code[filename] = new_code
                with open(join(self.path, filename), "w", encoding="utf-8") as prog:
                    prog.write(new_code)
        str_state = self.get_type_state_key()
        if str_state in self.seen_states:
            print("Tested before.....")
            return self.seen_states[str_state]
        self.seen_states[str_state] = False
        cmds = "source /home/gsakkas/.ghcup/env; " # for local Haskell installation
        cmds += "export PATH=$PATH:/home/gsakkas/usr/bin; " # for local Z3 installation
        cmds += f"cd {self.exec_path}; "
        cmds += f"stack build"
        # cmds += f"git restore src"
        test_output = subp.run(cmds, shell=True, check=False, capture_output=True)
        result = test_output.stderr.decode('utf-8').strip()
        # print("<<" * 42)
        # print("/home/gsakkas/hsalsa20".join(result.split("/home/gsakkas/hsalsa20")[1:]))
        # print(">>" * 42)
        if result != "" and "UNSAFE" not in result and "Error:" not in result and "SAFE" in result:
            self.seen_states[str_state] = True
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
        return self.file_obj.total_times_tested[self.func]
        # key = (self.filename, self.func)
        # all_runs = self.file_obj.total_times_tested[self.func]
        # for dep_filename, dep_func in self.dependencies[key]:
        #     dep_file_obj = self.files[dep_filename]
        #     all_runs += dep_file_obj.total_times_tested[dep_func]
        # return all_runs // len(self.file_obj.type_preds_cache[self.func])


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


def run_tests(path, args):
    dependencies = {}
    prompt_cache = {}
    with open("./benchmarks/hsalsa20_dependencies_with_initialization_and_indirect_deps.json", "r", encoding="utf-8") as dep_file:
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
            total_num_of_funcs_per_file[filename] += 1
            total_num_of_correct_funcs[filename] = 0
            curr_num_of_funcs_tested[filename] = 0
            runs_upper_bound[(filename, func)] = 30 if len(new_dependencies[(filename, func)]) < 2 else 20
    dependencies = new_dependencies
    # NOTE: Clean-up in case we are missing some file-function pairs?
    for filename, func in dependencies:
        deps = dependencies[(filename, func)]
        dependencies[(filename, func)] = [t for t in deps if t in deps]

    # Load (finetuned) code LLM
    code_llm = StarCoderModel()

    # Initialize Liquid Haskell project state with all the files and function dependencies
    project_state = ProjectState(path, args.exec_dir, all_files, dependencies, args.max_preds)

    total_num_of_progs = len(project_state.files)
    num_of_iterations = 0
    num_of_skipped = 0
    total_llm_calls = 0
    func_stack = list(reversed(project_state.get_all_file_func_pairs()))
    MAX_ITERATIONS = len(func_stack) * (args.max_preds * 2)
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
        if num_of_skipped > len(func_stack):
            print("Something went wrong... No more deps could be tested...")
            print("The remaining stack:")
            print(f"{filename} --> {func})", flush=True)
            for _fname, _func in reversed(func_stack):
                print(f"{_fname} --> {_func}")
            break
        print("=" * 42)
        print(f"Solving {filename} ({func})...", flush=True)
        if not project_state.deps_have_been_proven():
            # Not all dependencies have a type, thus no need to check this key yet
            num_of_skipped += 1
            print(f"Not all dependencies are done yet... Skipping...")
            func_stack.append(key)
            # Add least tested dependency to stack
            next_key = project_state.get_least_tested_dependency_not_in_stack(func_stack)
            if next_key:
                func_stack.append(next_key)
                print(f"Backtracking to dependent function = {next_key}...", flush=True)
            else:
                print("No more dependent functions to backtrack to...", flush=True)
                print("Pushing failed function at the beginning of stack...", flush=True)
                # Removing current key
                func_stack.pop()
                func_stack.appendleft(key)
            continue
        print_stack(func_stack)
        num_of_skipped = 0
        num_of_iterations += 1
        solved = False
        if not project_state.is_using_ground_truth() and project_state.state_llm_calls_less_than_max() and \
                (file_obj.tested_types_num[func] == 0 or not project_state.has_more_preds_avail()) :
            prompt = file_obj.make_prompt_for_func(func)
            # Prompt key includes all predicted types so far, the current file and function and how many times we called with this prompt
            prompt_key = filename + "--" + func + "<-->" + project_state.get_type_state_key() + "<-->" + str(file_obj.num_of_llm_calls[func] // args.total_preds)
            if prompt_key not in prompt_cache:
                prompt_cache[prompt_key] = get_type_predictions(prompt, filename, func, file_obj.ground_truths[func], code_llm, args)
                with open(args.cache_file, "w", encoding="utf-8") as cache_file:
                    cache_file.write(json.dumps(prompt_cache, indent=4))
            mtype_preds = prompt_cache[prompt_key]
            total_llm_calls += 1
            num_of_preds = args.total_preds
            if file_obj.type_preds_cache[func]:
                prev_preds = file_obj.type_preds_cache[func]
                mtype_preds.extend(prev_preds)
                mtype_preds = list(set(mtype_preds))
                num_of_preds += file_obj.num_of_llm_calls[func]
                # NOTE: Exit criteria for the backtracking loop
                # If the LLM can't generate any new types, then try the ground truth type or go back
                # NOTE: Or if LLM generates too many different types (heuristic), probably they are not that good
                if len(mtype_preds) == len(file_obj.type_preds_cache[func]) and dependencies[key]:
                    file_obj.type_preds_cache[func] = mtype_preds
                    file_obj.num_of_llm_calls[func] = num_of_preds
                    print("No new predictions...", flush=True)
                    # project_state.clean_func_dependants_and_add_to_stack(func_stack)
                    # # Add failed function to retry later
                    # func_stack.append(key)
                    # project_state.clean_func()
                    # # Add least tested dependency to stack
                    # next_key = project_state.get_least_tested_dependency_not_in_stack(func_stack)
                    # if next_key:
                    #     func_stack.append(next_key)
                    #     print(f"Backtracking to dependent function = {next_key}...", flush=True)
                    # else:
                    #     print("No more dependent functions to backtrack to...", flush=True)
                    #     print("Pushing failed function at the beginning of stack...", flush=True)
                    #     # Removing current key
                    #     func_stack.pop()
                    #     func_stack.appendleft(key)
                    # continue
                elif len(mtype_preds) == len(file_obj.type_preds_cache[func]):
                    file_obj.type_preds_cache[func] = mtype_preds
                    file_obj.num_of_llm_calls[func] = num_of_preds
                    print("No new predictions...", flush=True)
                    # project_state.clean_func_dependants_and_add_to_stack(func_stack)
                    # # Add failed function to retry later
                    # func_stack.append(key)
                    # project_state.clean_func()
                    # # Add a random next key since we don't have any dependencies to check
                    # # NOTE: This should happen very rarely or not at all
                    # all_keys = project_state.get_all_file_func_pairs()
                    # # shuffle(all_keys)
                    # next_key, i = all_keys[0], 0
                    # next_filename, next_func = next_key
                    # next_file_obj = project_state.files[next_filename]
                    # # while (next_key in func_stack or next_file_obj.current_types[next_func] or next_file_obj.using_ground_truth[next_func]) and i + 1 < len(all_keys):
                    # while (next_key in func_stack or next_file_obj.current_types[next_func]) and i + 1 < len(all_keys):
                    #     i += 1
                    #     next_key = all_keys[i]
                    #     next_filename, next_func = next_key
                    #     next_file_obj = project_state.files[next_filename]
                    # if i + 1 < len(all_keys):
                    #     func_stack.append(next_key)
                    #     print(f"Backtracking to random function = {next_key}...", flush=True)
                    # else:
                    #     print("No more available functions to backtrack to...", flush=True)
                    #     print("Pushing failed function at the beginning of stack...", flush=True)
                    #     # Removing current key
                    #     func_stack.pop()
                    #     func_stack.appendleft(key)
                    # continue
                elif (len(mtype_preds) > 20 or len(mtype_preds) == len(file_obj.type_preds_cache[func]) or len(mtype_preds) == 0) and not project_state.is_using_ground_truth():
                    print(f"Testing the ground truth type, since we got too many unique or no predictions ({len(mtype_preds)})...", flush=True)
                    project_state.set_file_func_to_ground()
            elif len(mtype_preds) < 5:
                # Prompt key includes all predicted types so far, the current file and function and how many times we called with this prompt
                prompt_key = filename + "--" + func + "<-->" + project_state.get_type_state_key() + "<-->" + str((file_obj.num_of_llm_calls[func] + args.total_preds) // args.total_preds)
                if prompt_key not in prompt_cache:
                    prompt_cache[prompt_key] = get_type_predictions(prompt, filename, func, file_obj.ground_truths[func], code_llm, args)
                    with open(args.cache_file, "w", encoding="utf-8") as cache_file:
                        cache_file.write(json.dumps(prompt_cache, indent=4))
                mtype_preds.extend(prompt_cache[prompt_key])
                total_llm_calls += 1
                mtype_preds = list(set(mtype_preds))
                num_of_preds += args.total_preds
            if not project_state.is_using_ground_truth():
                file_obj.type_preds_cache[func] = mtype_preds
                file_obj.num_of_llm_calls[func] = num_of_preds
        elif not project_state.is_using_ground_truth() and not project_state.state_llm_calls_less_than_max() and not project_state.has_more_preds_avail():
            # If we can't generate any good types with LLMs, then test the ground truth (correct type from user)
            print(f"Testing the ground truth type, since we reached max limit of predictions...", flush=True)
            project_state.set_file_func_to_ground()

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
                # Just a random check, cause LH crashes for too long types
                if len(type_prediction) > len(file_obj.ground_truths[func]) + 64:
                    print("...UNSAFE")
                    type_prediction = project_state.get_next_pred()
                    continue
                if project_state.get_times_tested_per_pred() >= runs_upper_bound[key]:
                    print("Too many failures for this type; Testing the ground truth type...")
                    project_state.set_file_func_to_ground()
                    type_prediction = file_obj.ground_truths[func]
                    print(f"Testing {{-@ {func} :: {type_prediction} @-}}...", flush=True)
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
            project_state.clean_func()
            # Add failed function to retry later
            func_stack.append(key)
            # Add least tested dependency to stack
            next_key = project_state.get_least_tested_dependency_not_in_stack(func_stack)
            if next_key:
                print(f"Backtracking to dependent function = {next_key}...", flush=True)
                func_stack.append(next_key)
            else:
                print("No more dependent functions to backtrack to...", flush=True)
                print("Pushing failed function at the beginning of stack...", flush=True)
                # Removing current key
                func_stack.pop()
                func_stack.appendleft(key)
        elif not project_state.is_using_ground_truth() and project_state.get_times_tested_per_pred() < runs_upper_bound[key]:
            # print("Pushing failed function at the beginning of stack...", flush=True)
            print(f"Trying again function {key}...", flush=True)
            project_state.clean_func_dependants_and_add_to_stack(func_stack)
            project_state.clean_func()
            func_stack.append(key)
        else:
            project_state.clean_func_dependants_and_add_to_stack(func_stack)
            project_state.clean_func()
            # Add failed function to retry later
            func_stack.appendleft(key)
            # # Add a random next key since we don't have any dependencies to check
            # # NOTE: This should happen very rarely or not at all
            # all_keys = project_state.get_all_file_func_pairs()
            # # shuffle(all_keys)
            # next_key, i = all_keys[0], 0
            # next_filename, next_func = next_key
            # next_file_obj = project_state.files[next_filename]
            # # while (next_key in func_stack or next_file_obj.current_types[next_func] or next_file_obj.using_ground_truth[next_func]) and i + 1 < len(all_keys):
            # while (next_key in func_stack or next_file_obj.using_ground_truth[next_func]) and i + 1 < len(all_keys):
            #     i += 1
            #     next_key = all_keys[i]
            #     next_filename, next_func = next_key
            #     next_file_obj = project_state.files[next_filename]
            # if i + 1 < len(all_keys):
            #     print(f"Backtracking to random function = {next_key}...", flush=True)
            #     func_stack.append(next_key)
            # else:
            #     print("No more available functions to backtrack to...", flush=True)
            #     print("Pushing failed function at the beginning of stack...", flush=True)
            #     # Removing current key
            #     func_stack.pop()
            #     func_stack.appendleft(key)

    total_ground_truths = {}
    total_num_of_ground_truths_used = 0
    total_num_of_refinemet_types = 0
    for filename, file_obj in project_state.files.items():
        given_by_user = 0
        for func in file_obj.liquid_funcs:
            if (filename, func) not in project_state.dependencies:
                given_by_user += 1
        total_ground_truths[filename] = sum(file_obj.using_ground_truth.values()) - given_by_user
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
    print(f"{num_of_iterations} loop iterations (i.e. total number of types checked)")
    # NOTE: batch_size comes from get_starcoder_code_suggestions.py parameter
    batch_size = 2
    print(f"{total_llm_calls * args.total_preds // batch_size} llm calls of {batch_size} type predictions")


if __name__ == "__main__":
    cmd_args = get_args()
    run_tests("../hsalsa20/src", cmd_args)