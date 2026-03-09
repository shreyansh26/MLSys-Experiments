SAFETENSORS_FAST_GPU=1 vllm serve \
    MiniMaxAI/MiniMax-M2.1 --trust-remote-code \
    --enable_expert_parallel --tensor-parallel-size 8 \
    --enable-auto-tool-choice --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think 
