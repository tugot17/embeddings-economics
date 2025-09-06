import sglang as sgl
import torch
import torch.cuda.nvtx as nvtx

if __name__ == '__main__':
    llm = sgl.Engine(model_path="Qwen/Qwen3-Embedding-8B", is_embedding=True)
    BATCH_SIZE = 1
    prompts = [f"Your text string goes here {i}"*1000 for i in range(BATCH_SIZE)]

    # print("Warming up...")
    # for i in range(5):
    #     _ = llm.encode(prompts)
    # torch.cuda.synchronize()

    print(f"Profiling SGLang embedding with batch_size={BATCH_SIZE}")
    print(f"Number of prompts: {len(prompts)}")

    # Start profiling
    torch.cuda.cudart().cudaProfilerStart()

    # Profile the embedding operation
    nvtx.range_push("sglang_embedding")
    embeddings = llm.encode(prompts)
    nvtx.range_pop()

    # Ensure all GPU work is complete
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()