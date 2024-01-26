# import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(
    "bigcode/starcoderbase-3b",
    cache_dir="/tmp3/gsakkas/huggingface"
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    "bigcode/starcoderbase-3b",
    device_map="auto",
    torch_dtype=torch.float16,
    # load_in_8bit=True,
    # low_cpu_mem_usage=True,
    cache_dir="/tmp3/gsakkas/huggingface"
)

model = PeftModel.from_pretrained(model, "/tmp3/gsakkas/checkpoints_the_stack/final_checkpoint") # final checkpoint
model = model.merge_and_unload()

# print(model._config)

def get_starcoder_code_suggestions(prompt, num_sugg):
    responses = []
    while len(responses) < num_sugg:
        local_responses = get_starcoder_response_with_retries(prompt, 3, 256)
        if not local_responses:
            return responses
        responses.extend(local_responses)
    return [extract_code_from_starcoder_suggestion(prompt, response) for response in responses[:num_sugg]]


def extract_code_from_starcoder_suggestion(prompt, code):
    """Extract the Python code from the StarCoder suggestion"""
    new_code = code.replace('\r\n', '\n')
    new_code = new_code.split("<fim_middle>")[1].rstrip().split("<|endoftext|>")[0].rstrip()
    new_code = '\n'.join(new_code.split('\n'))
    # print(f"extracted code from \n'''{code}'''\n is \n'''{new_code}'''\n")
    return new_code


def get_starcoder_response_with_retries(prompt, num_seqs, max_new_tokens):
    """Query HuggingFace StarCoder with retries"""
    local_num_seqs = min(10, num_seqs)
    for _ in range(2):
        try:
            input_ids = tokenizer([prompt], return_tensors="pt", padding=True).input_ids
            input_ids = input_ids.to('cuda')

            max_length = min(2 * input_ids.flatten().size(0), input_ids.flatten().size(0) + max_new_tokens)
            if max_length > 8192:
                print(f"Warning: max_length {max_length} is greater than the context window 8192")

            predictions = model.generate(input_ids=input_ids,
                                        pad_token_id=tokenizer.eos_token_id,
                                        do_sample=True,
                                        temperature=0.98,
                                        top_k=100,
                                        top_p=0.95,
                                        num_return_sequences=local_num_seqs,
                                        max_length=max_length)
            responses = [tokenizer.decode(pr, clean_up_tokenization_spaces=False) for pr in predictions]
            return responses
        except Exception as err:
            print(f"Exception in get_starcoder_response_with_retries {err}...||...")
            # traceback.print_tb(err.__traceback__)
            continue
    return None
