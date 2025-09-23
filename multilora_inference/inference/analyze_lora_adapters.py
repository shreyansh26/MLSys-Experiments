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

def analyze_adapter_weights(checkpoint_path: str, base_model_config: AutoConfig):
    lora_adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    tensors = {}
    with safe_open(lora_adapter_path, framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    print(tensors.keys())

    modules_with_lora = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    A_weights = []
    B_weights = []

    counter = {}
    A_weights = {}
    B_weights = {}

    for module in modules_with_lora:
        A_weights[module] = []
        B_weights[module] = []
        counter[module] = 0
        for k in tensors.keys():
            if module in k:
                counter[module] = counter.get(module, 0) + 1
    
    print("Found LoRA adapters for the following modules:", counter)
    print("Base model hidden size:", base_model_config.hidden_size)
    print("Base model num hidden layers:", base_model_config.num_hidden_layers)


    print("A:", tensors[f"base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"].shape)
    print("A:", tensors[f"base_model.model.model.layers.1.mlp.down_proj.lora_A.weight"].shape)

    print("="*100)
    print("Shape of LoRA adapters:")
    for layer in range(base_model_config.num_hidden_layers):
        for module in modules_with_lora:
            if module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                module_name = "self_attn." + module
            elif module in ["up_proj", "gate_proj", "down_proj"]:
                module_name = "mlp." + module

            A_weights[module].append(tensors[f"base_model.model.model.layers.{layer}.{module_name}.lora_A.weight"])
            B_weights[module].append(tensors[f"base_model.model.model.layers.{layer}.{module_name}.lora_B.weight"])

            if layer == 0:
                print("Module:", module)
                print("A:", tensors[f"base_model.model.model.layers.{layer}.{module_name}.lora_A.weight"].shape)
                print("B:", tensors[f"base_model.model.model.layers.{layer}.{module_name}.lora_B.weight"].shape)

    print("="*100)
    print("Combined shape of A and B weights:")
    for module in modules_with_lora:
        print("Module:", module)
        A_weights[module] = torch.stack(A_weights[module], dim=0)
        B_weights[module] = torch.stack(B_weights[module], dim=0)
        print("A:", A_weights[module].shape)
        print("B:", B_weights[module].shape)

if __name__ == "__main__":
    checkpoint_path = "/mnt/ssd2/shreyansh/models/multilora/ifeval_like_data_2025-09-23_08:26:56/epoch-2"
    base_model_config = get_base_model_config(checkpoint_path)
    analyze_adapter_weights(checkpoint_path, base_model_config)
