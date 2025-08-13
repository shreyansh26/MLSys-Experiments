import json
import torch

if __name__ == "__main__":
    model_name = "llama_3b_instruct"
    model_path = f"./{model_name}/original"
    model_config = f"{model_path}/params.json"
    with open(model_config, "r") as f:
        params = json.load(f)
    print(params)
    print(params["dim"])
    print(params["n_layers"])
    print(params["n_heads"])
    print(params["n_kv_heads"])