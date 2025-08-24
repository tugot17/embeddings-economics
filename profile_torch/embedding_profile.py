import torch
from vllm import LLM
from text_generator import TextGenerator
import torch.cuda.nvtx as nvtx

# Initialize
text_generator = TextGenerator()
model = LLM(model="Qwen/Qwen3-Embedding-4B", task="embed", enforce_eager=True, max_model_len=8192, dtype=torch.bfloat16)

batch_size = 128
prompts = text_generator.generate_batch(batch_size=batch_size, length_distribution="short")

# Warmup run (important for accurate profiling)
print("Warming up...")
_ = model.embed(prompts[:2])  # Small warmup
torch.cuda.synchronize()

print(f"Profiling vLLM embedding with batch_size={batch_size}")
print(f"Number of prompts: {len(prompts)}")

# Start profiling
torch.cuda.cudart().cudaProfilerStart()

# Profile the embedding operation
nvtx.range_push("vllm_embedding")
embeddings = model.embed(prompts)
nvtx.range_pop()

# Ensure all GPU work is complete
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
