import os
import random
import sys
import time
import timeit

import openai

# FIXME
# FIXME Probably outdated for the new experiments
# FIXME (have to update to use gpt-4 probably)

# OpenAI key
OPENAI_KEY = 'OPENAI_KEY'
if OPENAI_KEY not in os.environ:
    print("Please set the environment variable OPENAI_KEY")
    sys.exit(1)
openai.api_key = os.environ[OPENAI_KEY]


def get_codex_code_suggestions(prompt, suffix, num_sugg):
    """Get code suggestions from Codex"""
    sampling_temperature = 0.6
    best_of = 5
    response, response_len = get_codex_response_with_retries(prompt, suffix, best_of, sampling_temperature, False, num_sugg)
    if response_len < 0:
        return []
    return [r['text'] for r in response['choices']]


def get_codex_code_suggestions_from_chat(prompt, num_of_tokens, num_sugg):
    """Get code suggestions from Codex chat"""
    sampling_temperature = 0.6
    best_of = num_sugg
    response, response_len = get_codex_chat_response_with_retries(prompt, num_of_tokens + 4, best_of, sampling_temperature, False, num_sugg)
    if response_len < 0:
        return []
    return [extract_code_from_codex_chat_suggestion(prompt, r['text']) for r in response['choices']]


def extract_code_from_codex_chat_suggestion(prompt, code):
    """Extract the Python code from the Codex chat suggestion"""
    new_code = code.replace('\r\n', '\n')
    new_code = new_code.strip()
    new_code = new_code.replace("\n ##", "\n###")
    new_code = new_code.replace("\n  ##", "\n###")
    new_code = new_code.replace("\n   ##", "\n###")
    new_code = new_code.replace("\n    ##", "\n###")
    new_code = new_code.replace("\n ###", "\n###")
    new_code = new_code.replace("\n  ###", "\n###")
    new_code = new_code.replace("\n   ###", "\n###")
    new_code = new_code.replace("\n    ###", "\n###")
    new_code = new_code.split("\n###")[0].strip()
    num_of_orig_lines = len(prompt.split('\n'))
    new_code = '\n'.join(new_code.split('\n')[:num_of_orig_lines])
    # print(f"extracted code from \n'''{code}'''\n is \n'''{new_code}'''\n")
    return new_code


def get_codex_response_with_retries(prompt, suffix, best_of, temp, echo, num_sugg):
    """Query OpenAI Codex with retries"""
    sleep_time = 60 # seconds
    local_num_sugg = 10
    local_best_of = local_num_sugg
    response = {'choices': []}
    response_len = -1
    # prev_current = -1
    token_limits_reached = 0
    for _ in range(num_sugg + 2):
        try:
            while response_len < num_sugg:
                response['choices'].extend(get_codex_response(prompt, suffix, local_best_of, temp, echo, local_num_sugg, 128, 'code-davinci-002')['choices'])
                response_len = min(len(response['choices']), num_sugg)
            return response, response_len
        except openai.error.RateLimitError as err:
            err_msg = str(err)
            limit = -1
            # current = -1
            err_type = "unknown"
            if "requests per min" in err_msg:
                limit   = int(err_msg.split("Limit: ")[1].split(" / min")[0])
                # current = int(err_msg.split("Current: ")[1].split(" / min")[0])
                err_type = "requests per min"
            elif "tokens per min" in err_msg:
                limit   = int(err_msg.split("Limit: ")[1].split(" / min")[0])
                # current = int(err_msg.split("Current: ")[1].split(" / min")[0])
                err_type = "tokens per min"
                token_limits_reached += 1
            # print(f"Rate limit reached ({current}/{limit} {err_type})")
            print(f"Rate limit reached ({limit} {err_type})")
            if err_type == "tokens per min":
                local_best_of = local_num_sugg = (local_num_sugg + 1) // 2
                # NOTE: OpenAI removed the "current" usage values
                # if token_limits_reached > 2 and local_num_sugg == 1 and current == prev_current:
                #     break
                # if token_limits_reached > 1 and local_num_sugg > 1 and current == prev_current:
                #     local_best_of = local_num_sugg = 1
                # else:
                #     local_num_sugg = max(int(local_num_sugg // (current / limit)), 1)
                #     local_best_of = local_num_sugg * 2
                # prev_current = current
            time.sleep(sleep_time + random.randint(1, 10))
            continue
        except openai.error.OpenAIError as err:
            print(f"Exception in get_codex_response_with_retries {err}")
            time.sleep(sleep_time)
            continue
        except Exception as err:
            print(f"Exception in get_codex_response_with_retries {err}")
            time.sleep(sleep_time)
            continue
    return None



def get_codex_chat_response_with_retries(prompt, max_tokens, best_of, temp, echo, num_sugg):
    """Query OpenAI Codex with retries"""
    sleep_time = 60 # seconds
    local_num_sugg = 10
    local_best_of = local_num_sugg
    local_prompt = prompt
    response = {'choices': []}
    response_len = -1
    # prev_current = -1
    token_limits_reached = 0
    for _ in range(num_sugg + 2):
        try:
            while response_len < num_sugg:
                response['choices'].extend(get_codex_chat_response(local_prompt, local_num_sugg, temp, echo, local_num_sugg, max_tokens, 'code-davinci-002')['choices'])
                response_len = min(len(response['choices']), num_sugg)
            return response, response_len
        except openai.error.RateLimitError as err:
            err_msg = str(err)
            limit = -1
            # current = -1
            err_type = "unknown"
            if "requests per min" in err_msg:
                limit   = int(err_msg.split("Limit: ")[1].split(" / min")[0])
                # current = int(err_msg.split("Current: ")[1].split(" / min")[0])
                err_type = "requests per min"
            elif "tokens per min" in err_msg:
                limit   = int(err_msg.split("Limit: ")[1].split(" / min")[0])
                # current = int(err_msg.split("Current: ")[1].split(" / min")[0])
                err_type = "tokens per min"
                token_limits_reached += 1
            # print(f"Rate limit reached ({current}/{limit} {err_type})")
            print(f"Rate limit reached ({limit} {err_type})")
            if err_type == "tokens per min":
                local_best_of = local_num_sugg = (local_num_sugg + 1) // 2
                # NOTE: OpenAI removed the "current" usage values
                # if token_limits_reached > 2 and local_num_sugg == 1 and current == prev_current:
                #     break
                # if token_limits_reached > 1 and local_num_sugg > 1 and current == prev_current:
                #     local_num_sugg = 1
                #     # Hacky way to reduce prompt length; it's based on the prompt message
                #     # FIXME: Has to change every time the message changes in the main script
                #     before = "Error in the following Buggy Python 3 Program\n"
                #     after = "### Python 3 Error message\n"
                #     tokens_to_remove = (current - limit) // 3
                #     local_prompt = local_prompt.split(before)[1].split(after)[0]
                #     num_removed_tokens = 0
                #     # Remove lines from the end of the program
                #     while num_removed_tokens < tokens_to_remove:
                #         local_prompt = '\n'.join(local_prompt.split('\n')[:-1])
                #         num_removed_tokens += len(local_prompt.rsplit('\n', maxsplit=1)[-1].split()) + 1
                # else:
                #     local_num_sugg = max(int(local_num_sugg // (current / limit)), 1)
                # prev_current = current
            time.sleep(sleep_time + random.randint(1, 10))
            continue
        except openai.error.OpenAIError as err:
            print(f"Exception in get_codex_response_with_retries {err}")
            time.sleep(sleep_time)
            continue
        except Exception as err:
            print(f"Exception in get_codex_response_with_retries {err}")
            time.sleep(sleep_time)
            continue
    return response, response_len


def get_codex_response(prompt, suffix, best_of, temp, echo, num_sugg, max_tokens, engine):
    assert best_of >= num_sugg
    response = openai.Completion.create(prompt=prompt,
                                        suffix=suffix,
                                        best_of=best_of,
                                        temperature=temp,
                                        top_p=0.95,
                                        echo=echo,
                                        max_tokens=max_tokens,
                                        # stop=["\nclass", "\ndef", "\n#", "\nif"],
                                        engine=engine,
                                        n=num_sugg)
    return response


def get_codex_chat_response(prompt, best_of, temp, echo, num_sugg, max_tokens, engine):
    assert best_of >= num_sugg
    response = openai.Completion.create(prompt=prompt,
                                        best_of=best_of,
                                        temperature=temp,
                                        top_p=0.95,
                                        echo=echo,
                                        max_tokens=max_tokens,
                                        stop=["\n'''", "\n\"\"\"", "\n##", "\n###"],
                                        engine=engine,
                                        n=num_sugg)
    return response
