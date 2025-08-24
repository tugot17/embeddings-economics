VLLM_TORCH_PROFILER_DIR=./traces/ python -m vllm.entrypoints.openai.api_server \
  --model "Qwen/Qwen3-Embedding-4B" \
  --task embed \
  --enforce-eager \
  --max-model-len 8192 \
  --dtype bfloat16