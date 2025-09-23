import os
import json
import torch
from safetensors import safe_open
from transformers import AutoConfig


def get_base_model_config(checkpoint_path: str):
    path = os.path.join(checkpoint_path, "adapter_config.json")
    config = json.load(open(path))
    base_model_name = config['base_model_name_or_path']
    base_model_config = AutoConfig.from_pretrained(base_model_name)
    return base_model_config

def load_adapter_weights(checkpoint_path: str, base_model_config: AutoConfig):
    lora_adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    tensors = {}
    with safe_open(lora_adapter_path, framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    print(tensors.keys())

    modules_with_lora = ["k_proj","up_proj","down_proj","q_proj","v_proj","o_proj","gate_proj"]

    counter = {}
    for module in modules_with_lora:
        counter[module] = 0
        for k in tensors.keys():
            if module in k:
                counter[module] = counter.get(module, 0) + 1
    print("Found LoRA adapters for the following modules:", counter)
    print("Base model hidden size:", base_model_config.hidden_size)
    print("Base model num hidden layers:", base_model_config.num_hidden_layers)

    print("="*100)
    print("Shape of LoRA adapters:")
    for module in modules_with_lora:
        print("Module:", module)
        if module in ["k_proj", "q_proj", "v_proj", "o_proj"]:
            module = "self_attn." + module
        elif module in ["up_proj", "down_proj", "gate_proj"]:
            module = "mlp." + module
        
        print("A:", tensors[f"base_model.model.model.layers.0.{module}.lora_A.weight"].shape)
        print("B:", tensors[f"base_model.model.model.layers.0.{module}.lora_B.weight"].shape)

if __name__ == "__main__":
    checkpoint_path = "/mnt/ssd2/shreyansh/models/multilora/ifeval_like_data_2025-09-23_08:26:56/epoch-2"
    base_model_config = get_base_model_config(checkpoint_path)
    load_adapter_weights(checkpoint_path, base_model_config)
