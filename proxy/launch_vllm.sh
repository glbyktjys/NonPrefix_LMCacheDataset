tmux new -s vllm

CUDA_VISIBLE_DEVICES=0,1,3,4 \
VLLM_USE_FLASHINFER_MOE_FP8=0 \
    vllm serve MiniMaxAI/MiniMax-M2.5 \
    --trust-remote-code \
    --tool-call-parser minimax_m2 \
    --enable-auto-tool-choice \
    --tensor-parallel-size 4 \
    --max-model-len 196608 \
    --port 8200

Ctrl-b d

# Later, reattach to see logs:
tmux attach -t vllm
tmux ls

# Kill the tmux session when you want to stop vLLM:
tmux kill-session -t vllm