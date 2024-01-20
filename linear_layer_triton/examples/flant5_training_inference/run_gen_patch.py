import time
import torch
from transformers import AutoTokenizer
from misc.patch_model import patch_model
from transformers import T5ForConditionalGeneration

import argparse

parser = argparse.ArgumentParser(description='FlanT5 generation')
parser.add_argument('--mode', type=str, default='pure_torch', choices=['pure_torch', 'triton_linear'], help='Model mode')

args = parser.parse_args()

NUM_ITERS = 10
MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

model = model.to('cuda')
model = model.half()
model.eval()

if args.mode == "triton_linear":
    patch_model(model.encoder)
    patch_model(model.decoder)

input_ids = tokenizer(["summarize: studies have shown that owning a dog is good for you."]*10, return_tensors="pt", padding='max_length', max_length=512, truncation=True).input_ids.to("cuda")  # Batch size 1

print(input_ids.shape)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print("Starting compilation and warmup!")
for _ in range(NUM_ITERS):
    outputs = model.generate(input_ids, max_new_tokens=200)
print("Done compilation and warmup!")

input_ids = tokenizer(["summarize: I am happy because I ate chocolate today. Chocolate is so good for you. Or is it?"]*10, return_tensors="pt", padding='max_length', max_length=512, truncation=True).input_ids.to("cuda")  # Batch size 1
start.record()
outputs = model.generate(input_ids, max_new_tokens=200)
end.record()
torch.cuda.synchronize()

print(f"Time taken - {start.elapsed_time(end)}ms")
print(f"Output - {tokenizer.decode(outputs[0], skip_special_tokens=True)}")