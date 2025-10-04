import os
import json
import torch
from safetensors import safe_open
from transformers import AutoConfig
from typing import List


def get_base_model_config(checkpoint_path: str):
    path = os.path.join(checkpoint_path, "adapter_config.json")
    adapter_config = json.load(open(path))
    base_model_name = adapter_config['base_model_name_or_path']
    base_model_config = AutoConfig.from_pretrained(base_model_name)
    return base_model_config, base_model_name, adapter_config

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

def get_A_B_weights(checkpoint_path: str, base_model_config: AutoConfig):
    lora_adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    tensors = {}
    with safe_open(lora_adapter_path, framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    
    modules_with_lora = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    A_weights = []
    B_weights = []

    A_weights = {}
    B_weights = {}

    for module in modules_with_lora:
        A_weights[module] = []
        B_weights[module] = []
    
    for layer in range(base_model_config.num_hidden_layers):
        for module in modules_with_lora:
            if module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                module_name = "self_attn." + module
            elif module in ["up_proj", "gate_proj", "down_proj"]:
                module_name = "mlp." + module

            A_weights[module].append(tensors[f"base_model.model.model.layers.{layer}.{module_name}.lora_A.weight"])
            B_weights[module].append(tensors[f"base_model.model.model.layers.{layer}.{module_name}.lora_B.weight"])

            # To handle no LoRA case
            A_weights[module].append(torch.zeros_like(tensors[f"base_model.model.model.layers.{layer}.{module_name}.lora_A.weight"]))
            B_weights[module].append(torch.zeros_like(tensors[f"base_model.model.model.layers.{layer}.{module_name}.lora_B.weight"]))

    for module in modules_with_lora:
        A_weights[module] = torch.stack(A_weights[module], dim=0)
        B_weights[module] = torch.stack(B_weights[module], dim=0)

    return A_weights, B_weights

def get_multilora_A_B_weights(checkpoint_path: List[str], base_model_config: AutoConfig, mode: str = "gbmm"):
    A_weights = {}
    B_weights = {}

    modules_with_lora = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]

    for module in modules_with_lora:
        A_weights[module] = []
        B_weights[module] = []

    for checkpoint in checkpoint_path:
        lora_adapter_path = os.path.join(checkpoint, "adapter_model.safetensors")
        tensors = {}
        with safe_open(lora_adapter_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k).cuda()

        A_weights_t = {}
        B_weights_t = {}

        for module in modules_with_lora:
            A_weights_t[module] = []
            B_weights_t[module] = []
        
        for layer in range(base_model_config.num_hidden_layers):
            for module in modules_with_lora:
                if module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    module_name = "self_attn." + module
                elif module in ["up_proj", "gate_proj", "down_proj"]:
                    module_name = "mlp." + module

                A_weights_t[module].append(tensors[f"base_model.model.model.layers.{layer}.{module_name}.lora_A.weight"])
                B_weights_t[module].append(tensors[f"base_model.model.model.layers.{layer}.{module_name}.lora_B.weight"])

        for module in modules_with_lora:
            A_weights[module].append(torch.stack(A_weights_t[module], dim=0))
            B_weights[module].append(torch.stack(B_weights_t[module], dim=0))

    if mode == "gbmm" or mode == "sgmv_triton":
        # No LoRA case
        for module in modules_with_lora:
            A_weights_t[module] = []
            B_weights_t[module] = []
        
        for layer in range(base_model_config.num_hidden_layers):
            for module in modules_with_lora:
                if module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    module_name = "self_attn." + module
                elif module in ["up_proj", "gate_proj", "down_proj"]:
                    module_name = "mlp." + module

                A_weights_t[module].append(torch.zeros_like(tensors[f"base_model.model.model.layers.{layer}.{module_name}.lora_A.weight"]))
                B_weights_t[module].append(torch.zeros_like(tensors[f"base_model.model.model.layers.{layer}.{module_name}.lora_B.weight"]))

        for module in modules_with_lora:
            A_weights[module].append(torch.stack(A_weights_t[module], dim=0))
            B_weights[module].append(torch.stack(B_weights_t[module], dim=0))

    if mode == "gbmm" or mode == "sgmv_triton":
        for module in modules_with_lora:
            A_weights[module] = torch.stack(A_weights[module], dim=0).transpose(0, 1).contiguous()
            B_weights[module] = torch.stack(B_weights[module], dim=0).transpose(0, 1).contiguous()

    elif mode.startswith("bgmv"):
        for module in modules_with_lora:
            A_weights[module] = torch.stack(A_weights[module], dim=0).transpose(0, 1)
            nlayers, nlora, r, d = A_weights[module].shape
            A_weights[module] = A_weights[module].reshape(-1, r, d).contiguous()
            B_weights[module] = torch.stack(B_weights[module], dim=0).transpose(0, 1)
            nlayers, nlora, r, d = B_weights[module].shape
            B_weights[module] = B_weights[module].reshape(-1, r, d).contiguous()

    return A_weights, B_weights

if __name__ == "__main__":
    checkpoint_path = "/mnt/ssd2/shreyansh/models/multilora/ifeval_like_data/epoch-2"
    base_model_config, base_model_name, adapter_config = get_base_model_config(checkpoint_path)
    # analyze_adapter_weights(checkpoint_path, base_model_config)

    adapter_names = ["ifeval_like_data", "multilingual_cohere_aya", "opc_evol_instruct", "text_to_sql", "infinity_instruct", "numina_math", "opc_sft_educational"]
    checkpoint_path_list = [f"/mnt/ssd2/shreyansh/models/multilora/{adapter_name}/epoch-2" for adapter_name in adapter_names]
    A_weights, B_weights = get_multilora_A_B_weights(checkpoint_path_list, base_model_config, mode="bgmv_cuda")
    print(A_weights.keys())
    print(B_weights.keys())

    print(A_weights["q_proj"].shape)
    print(B_weights["q_proj"].shape)