import argparse
import errno
import functools
import json
import os
import signal
import time
import traceback
from concurrent.futures import TimeoutError
from functools import partial
from os.path import exists, join

import deepspeed

import get_llm_code_suggestions as codex
from get_starcoder_code_suggestions import StarCoderModel

TIMEOUT = 60 * 30


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


def rate(secs, times):
    in_set = list(filter(lambda x: x <= secs, times))
    return len(in_set) * 100.0 / len(times)


@timeout(TIMEOUT)
def llm_instruct_prompt_prediction(orig_prg, cmd_args, llm, cache_key=None, cache=None):
    """ Use LLM to fix runtime error using chat prompt """
    candidate_progs = [orig_prg]
    seen_progs = []
    seen_progs.extend([orig_prg])
    seen_progs = set(seen_progs)
    list_of_repaired_progs = []
    for _ in range(cmd_args.max_cost):
        for prog in candidate_progs:
            prompt = prog
            key = orig_prg + "<<break>>" + prompt + "<<break>>" + cmd_args.llm + "<<break>>" + str(cmd_args.total_repairs) + "<<break>>" + str(cmd_args.total_repairs)
            if cache_key:
                key = cache_key
            if cmd_args.use_cache and key in cache and cache[key] != []:
                prog_repairs = cache[key]
            elif "starcoder" in cmd_args.llm:
                prog_repairs = llm.get_code_suggestions(prompt, cmd_args.total_repairs)
            else:
                prog_repairs = codex.get_codex_code_suggestions_from_chat(prompt, 256, cmd_args.total_repairs)
            list_of_repaired_progs.extend(prog_repairs)
            if (cmd_args.update_cache or cmd_args.create_cache_only):
                cache[key] = prog_repairs
            if list_of_repaired_progs is None or not list_of_repaired_progs:
                return None
            candidate_progs = list(set(list_of_repaired_progs) - seen_progs)
            seen_progs = seen_progs.union(set(list_of_repaired_progs))
    return list(set(list_of_repaired_progs))


def get_predictions(cmd_args, llm, cache, tup):
    dct = json.loads(tup)
    # NOTE: We don't want to keep track of Codex API waits, so we use `process_time`
    start_time = time.process_time()
    orig_bad = dct['bad'].rstrip()
    list_of_repaired_progs = llm_instruct_prompt_prediction(orig_bad, cmd_args, llm, dct['key'] if 'key' in dct else None, cache)
    run_time = time.process_time() - start_time

    return (run_time, list_of_repaired_progs)


def run_llm_generation(cmd_args):
    if "starcoder" in cmd_args.llm:
        code_llm = StarCoderModel()
    else:
        code_llm = None
    done = 0
    failed = 0
    dataset = []
    avg_run_time = 0.0
    repair_times = []
    dataset_part_file = join(cmd_args.data_dir, cmd_args.data_file)
    if exists(dataset_part_file):
        with open(dataset_part_file, "r", encoding="utf-8") as in_file:
            dataset = in_file.read().strip().split('\n')
    cache_file = join(cmd_args.data_dir, cmd_args.cache_file)
    cache = {}
    if (cmd_args.use_cache or cmd_args.update_cache or cmd_args.create_cache_only) and exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as cf:
            cache = json.loads(cf.read())
    # dataset = dataset[::2]
    print("# LH types to predict:", len(dataset))
    get_predictions_temp = partial(get_predictions, cmd_args, code_llm, cache)
    for sample in dataset:
        try:
            intermediate = get_predictions_temp(sample)
            if not intermediate:
                failed += 1
                continue
            run_time, _ = intermediate
            done += 1
            avg_run_time += run_time

            if cmd_args.create_cache_only:
                if (failed + done) % 1 == 0:
                    print(f"### Dataset size: {done} / {failed + done}")
                    print(f"# Mean repair time: {avg_run_time / done:.2f} sec")
                continue

            repair_times.append(run_time)
        except TimeoutError as _:
            print('Timer expired!')
            failed += 1
            if cmd_args.create_cache_only:
                if (failed + done) % 1 == 0:
                    print(f"### Dataset size: {done} / {failed + done}")
                    print(f"# Mean repair time: {avg_run_time / done:.2f} sec")
                continue
            # run_time = TIMEOUT
            # avg_run_time += run_time
            # repair_times.append(run_time)
        except Exception as err:
            print("WHY here?!", str(err))
            traceback.print_tb(err.__traceback__)
            failed += 1
            if cmd_args.create_cache_only:
                if (failed + done) % 1 == 0:
                    print(f"### Dataset size: {done} / {failed + done}")
                    print(f"# Mean repair time: {avg_run_time / done:.2f} sec")
                continue
            # run_time = TIMEOUT
            # avg_run_time += run_time
            # repair_times.append(run_time)

    if cmd_args.update_cache or cmd_args.create_cache_only:
        with open(cache_file, "w", encoding="utf-8") as cf:
            cf.write(json.dumps(cache, indent = 4))
    if cmd_args.create_cache_only:
        print(f"### Dataset size: {done} / {failed + done}")
        print(f"# Mean prediction time: {avg_run_time / done:.2f} sec")


def get_args():
    _parser = argparse.ArgumentParser(description='llm_repair')
    _parser.add_argument('-r', '--total_repairs', default=10, type=int,
                        help='total repairs to generate with the model (default: 10)')
    _parser.add_argument('-c', '--max_cost', default=1, type=int,
                        help='max repair cost (default: 1)')
    _parser.add_argument('--llm', default="codex",
                        help='llm to use for code generation {codex, incoder-1B, -6B, codegen25-7B, starcoder} (default: codex)')
    _parser.add_argument('--cache_file', default="",
                        help='use the given file for prompt -> generation cache (default: no cache)')
    _parser.add_argument('--update_cache', action='store_true',
                        help='update the prompt cache (default: False)')
    _parser.add_argument('--use_cache', action='store_true',
                        help='use the prompt cache (default: False)')
    _parser.add_argument('--create_cache_only', action='store_true',
                        help='only create the prompt cache and don\'t run any tests (default: False)')
    _parser.add_argument('--data_dir', default=".",
                        help='input data directory (default: .)')
    _parser.add_argument('--out_dir', default="./results",
                        help='output data directory (default: ./results)')
    _parser.add_argument('--data_file',
                        help='input data file (default: None)')
    _parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    _parser = deepspeed.add_config_arguments(_parser)
    _args = _parser.parse_args()
    return _args


if __name__ == "__main__":
    args = get_args()

    run_llm_generation(args)
