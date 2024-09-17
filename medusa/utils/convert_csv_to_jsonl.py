import pandas as pd
import json
import argparse
import transformers
import os

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,
                                                    use_auth_token=os.environ["HF_TOKEN"],
                                                    trust_remote_code=True)
tokenizer.model_max_length = 4096
print(f"tokenizer.model_max_length: {tokenizer.model_max_length}")

MAX_NEW_TOKENS = 512
MAX_LENGTH = tokenizer.model_max_length - MAX_NEW_TOKENS

def update_to_max_length(tokenizer, instruction_fn, prompt, conv, max_length=4096):
    instruction_token_ids = tokenizer(instruction_fn(prompt, conv)).input_ids
    instruction_len = len(instruction_token_ids)
    if instruction_len > max_length:
        # print(instruction_len)
        only_instruction_token_ids = tokenizer(instruction_fn(prompt, '')).input_ids
        # print('1: ', len(only_instruction_token_ids))
        only_instruction_len = len(only_instruction_token_ids) + 1
        conv_maxlen = max_length - only_instruction_len
        conv_token_ids = tokenizer(conv, truncation=True, max_length=conv_maxlen).input_ids
        conv_token_ids = conv_token_ids[1:]             # to avoid <s> token
        # print('2: ', len(conv_token_ids))
        conv = tokenizer.decode(conv_token_ids)
        # print(instruction_fn(prompt, conv))
    return instruction_fn(prompt, conv)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv_path", type=str)
    parser.add_argument("--output_jsonl_path", type=str)
    # parser.add_argument("--instruct", action=argparse.BooleanOptionalAction)
    parser.add_argument("--instruct", type=bool, default=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv_path)
    with open(args.output_jsonl_path, "w") as f:
        for idx, row in df.iterrows():
            prompt = row['prompt']
            input_text = row['input_text']
            output_text = row['output_text']
            output_text = " " + output_text
            if not args.instruct:
                instruction_fn = lambda p,conv: ("You are given a instruction a question or a task, and some data. You are required "
                                "to respond to the question or task by using the data. "
                                "The data is one of: (1) full or partial transcript of a conversation, "
                                "(2) a summary of a conversation.\n\n"
                                f"<instruction start>\n{p}\n<instruction end>\n\n"
                                f"<conversation data start>\n{conv}\n<conversation data end>"
                            )
            else:
                instruction_fn = lambda p,conv: ("[INST] Respond to this task based on the given full or partial "
                            f"conversation transcript: {p} \n"
                            f"Conversation Transcript:\n{conv}\n [/INST] "
                            )
            instruction = instruction_fn(prompt, input_text)
            if tokenizer:
                instruction = update_to_max_length(tokenizer, instruction_fn, prompt, input_text, max_length=MAX_LENGTH)

            new_row = {
                    "instruction" : instruction,
                    "output" : output_text
                    }
            f.write(json.dumps(new_row) + "\n")
