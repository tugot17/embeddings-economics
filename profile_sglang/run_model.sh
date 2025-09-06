export SGLANG_TORCH_PROFILER_DIR=./profile_log

CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Embedding-8B \
  --is-embedding \
  --host 0.0.0.0 \
  --port 8000 \
  # --attention-backend flashinfer
  

# VLLM_TORCH_PROFILER_DIR=./profile_log python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-Embedding-8B --port 8000 