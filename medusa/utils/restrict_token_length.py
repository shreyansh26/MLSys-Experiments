import pandas as pd
import argparse
from transformers import AutoTokenizer
from tqdm.auto import tqdm
tqdm.pandas()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3500, help="Number of max tokens in prompt, input_text, output_text combined")
    parser.add_argument("--input_csv_path", type=str)
    parser.add_argument("--output_csv_path", type=str)
    args = parser.parse_args()

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def len_func(row):
        return len(tokenizer(row["prompt"] + \
                    " " + row["input_text"] + \
                    " " + row["output_text"]).input_ids)

    df = pd.read_csv(args.input_csv_path)
    df["token_lens"] = df.progress_apply(len_func, axis=1)
    df = df.loc[df.token_lens <= args.limit]
    df.to_csv(args.output_csv_path, index=False)


