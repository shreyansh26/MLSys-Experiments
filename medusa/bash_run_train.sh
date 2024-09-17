export HF_TOKEN=abcd
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=105 --rdzv_endpoint="0.0.0.0:29507" train_instruct.py