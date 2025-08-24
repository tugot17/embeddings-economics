import torch
from vllm import LLM
from text_generator import TextGenerator
from torch.profiler import profile, record_function, ProfilerActivity

# Initialize
text_generator = TextGenerator()
model = LLM(model="Qwen/Qwen3-Embedding-4B", task="embed", enforce_eager=True, max_model_len=8192, dtype=torch.bfloat16)

batch_size = 128
prompts = text_generator.generate_batch(batch_size=batch_size, length_distribution="long")

# Warmup run
print("Warming up...")
_ = model.embed(prompts[:2])
torch.cuda.synchronize()

print(f"Profiling vLLM embedding with batch_size={batch_size}")
print(f"Number of prompts: {len(prompts)}")

# Profile with PyTorch profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True
) as prof:
    with record_function("vllm_embedding"):
        embeddings = model.embed(prompts)

# Export results
prof.export_chrome_trace("vllm_embedding_trace.json")
prof.export_stacks("vllm_embedding_stacks.txt", "self_cuda_time_total")

# Print summary
print("\n" + "="*80)
print("PYTORCH PROFILER SUMMARY")
print("="*80)

# Top CUDA operations
print("\nTop 10 CUDA operations by time:")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Memory usage
print("\nTop 10 operations by memory usage:")
print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

print("\nProfile files generated:")
print("- vllm_embedding_trace.json (view in chrome://tracing)")
print("- vllm_embedding_stacks.txt (stack traces)")