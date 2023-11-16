BATCH_SIZE = 4
SEQ_LEN = 64
SAVE_INTERVAL = 50
TRAIN_STEP = 100

from tqdm import tqdm
import os
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

# model = AutoModelForCausalLM.from_pretrained("gpt2")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
optimizer = Adam(model.parameters(), lr=3e-5)
tokenizer = AutoTokenizer.from_pretrained("t5-large")

# Add pad token for batch training because GPT2 tokenizer doesn't have pad token.
tokenizer.pad_token = tokenizer.eos_token

# model = defined in section 2.2
import oslo
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel import TensorParallel

tp_size = 2
tp_depth = 1
dp_size = 2

parallel_context = ParallelContext.from_torch(
    data_parallel_size=dp_size,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_1D,
    tensor_parallel_depth=tp_depth,
)
model = TensorParallel(model, parallel_context)
oslo.ready(model, parallel_context)

from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler

datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(_) for _ in datasets[: TRAIN_STEP * BATCH_SIZE]]
print(len(datasets))
print(len(set(datasets)))
print(len(set(datasets[::2])))
print(len(set(datasets[1::2])))

rank = parallel_context.get_local_rank(ParallelMode.DATA)

if rank == 0:
    with open('data.txt', 'w') as f:
        f.write(str(datasets))

# if rank == 0:
#     print(datasets[1::2])
    
train_sampler = DistributedSampler(
        datasets, num_replicas=dp_size, rank=rank
    )
dataloader = DataLoader(datasets, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=False)

d0 = []
d1 = []

print(len(dataloader))

for step, batch in enumerate(tqdm(dataloader)):
    optimizer.zero_grad()

    if rank == 1:
        d1.extend(batch)
    if rank == 0:
        d0.extend(batch)
    # Make batch
    input_batch = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SEQ_LEN,
    ).to("cuda")

    # Forward-Backward-Step
    loss = model(**input_batch, labels=input_batch["input_ids"]).loss
    loss.backward()
    optimizer.step()

# Save the merged model using `save_pretrained`
model.save_pretrained(
    save_directory="./parallel_ckpt",
    merge_checkpoints=True # Different point in Section 2.6
)

if rank == 0:
    with open('d0.txt', 'w') as f:
        f.write(str(d0))

if rank == 1:
    with open('d1.txt', 'w') as f:
        f.write(str(d1))