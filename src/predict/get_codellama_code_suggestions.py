import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class CodeLlamaModel():
    def __init__(self, cdir="/tmp3/gsakkas/huggingface"):
        tokenizer = AutoTokenizer.from_pretrained(
            "codellama/CodeLlama-7b-hf",
            cache_dir=cdir
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            "codellama/CodeLlama-7b-hf",
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
            cache_dir=cdir
        )

        self.tokenizer = tokenizer
        self.model = model


    def get_code_suggestions(self, prompt, num_sugg):
        responses = []
        while len(responses) < num_sugg:
            local_responses = self.get_response_with_retries(prompt, 1, 128)
            if not local_responses:
                return responses
            responses.extend(local_responses)
        return [self.extract_code_from_suggestion(response) for response in responses[:num_sugg]]


    def extract_code_from_suggestion(self, code):
        """Extract new code from the CodeLlama suggestion"""
        new_code = code.replace('\r\n', '\n')
        if "<MID>" in new_code:
            new_code = new_code.split("<MID>")[1].rstrip()
        if "@-}" in new_code:
            new_code = new_code.split("@-}")[0].rstrip()
        if "<EOT>" in new_code:
            new_code = new_code.split("<EOT>")[0].rstrip()
        if "<file_sep>" in new_code:
            new_code = new_code.split("<file_sep>")[0]

        # print(f"extracted code from \n'''{code}'''\n is \n'''{new_code}'''\n")
        return new_code


    def get_response_with_retries(self, prompt, num_seqs, max_new_tokens):
        """Query HuggingFace CodeLlama with retries"""
        local_num_seqs = min(20, num_seqs)
        for _ in range(2):
            try:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt", padding=True).to('cuda')

                max_length = input_ids.flatten().size(0) + max_new_tokens
                if max_length > 8192:
                    print(f"Warning: max_length {max_length} is greater than the context window 8192")

                predictions = self.model.generate(input_ids=input_ids,
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            do_sample=True,
                                            temperature=0.98,
                                            top_k=100,
                                            top_p=0.95,
                                            num_return_sequences=local_num_seqs,
                                            max_new_tokens=max_new_tokens)
                responses = [self.tokenizer.decode(pr, clean_up_tokenization_spaces=False) for pr in predictions]
                return responses
            except Exception as err:
                print(f"Exception in get_response_with_retries {err}...||...")
                local_num_seqs //= 2
                continue
        return None
