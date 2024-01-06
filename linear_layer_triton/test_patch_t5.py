import time
import torch
import copy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from patch_linear_layer import patch_linear_layer
# from torch.fx import symbolic_trace
from transformers.utils.fx import symbolic_trace

NUM_ITERS = 1000
MODEL_NAME = "google/flan-t5-base"
# MODEL_NAME = "t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model = model.to('cuda')
model = model.half()
gm = symbolic_trace(
    model,
    input_names=["input_ids", "labels"]) # "attention_mask", "decoder_input_ids"
gm_old = copy.deepcopy(gm)

patch_linear_layer(gm, debug=False)

print(gm_old.code)
print("**"*100)
print(gm.code)

input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids.to("cuda")
labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids.to("cuda")
outputs = model(input_ids=input_ids, labels=labels)
print(outputs)

model.forward = gm_old
for i in range(10):
    _ = model(input_ids=input_ids, labels=labels)

print("Warmup (for Torch) done!")

torch_time = 0
for i in range(NUM_ITERS):
    t1 = time.time_ns()
    torch_output = model(input_ids=input_ids, labels=labels)
    t2 = time.time_ns()
    torch_time += (t2 - t1)

torch_time = torch_time / NUM_ITERS / 1_000_000   

model.forward = gm
for i in range(10):
    _ = model(input_ids=input_ids, labels=labels)

print("Compilation (for Triton) done!")

triton_time = 0
for i in range(NUM_ITERS):
    t1 = time.time_ns()
    triton_output = model(input_ids=input_ids, labels=labels)
    t2 = time.time_ns()
    triton_time += (t2 - t1)

triton_time = triton_time / NUM_ITERS / 1_000_000   

print(f"Triton time: {triton_time}ms")
print(f"Torch time: {torch_time}ms")

try:
    torch.testing.assert_close(torch_output.loss, triton_output.loss, atol=1e-2, rtol=0)
    print("✅ Triton and Torch match!")
except Exception as e:
    print(e)
    print("❌ Triton and Torch differ!")

input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids.to("cuda")  # Batch size 1

model.forward = gm_old
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

model.forward = gm
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))