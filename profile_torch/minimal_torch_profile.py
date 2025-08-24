import torch

device = 'cuda:0'
x = torch.randn(1024*1024, device=device)

# Warmup
torch.square(x)
torch.cuda.synchronize()

# Profile target operations
torch.cuda.cudart().cudaProfilerStart()

# Profile square (memory-bound)
torch.cuda.nvtx.range_push("square_op")
result1 = torch.square(x)
torch.cuda.nvtx.range_pop()

# Profile sin (compute-bound)
torch.cuda.nvtx.range_push("sin_op")
result2 = torch.sin(x)
torch.cuda.nvtx.range_pop()

torch.cuda.cudart().cudaProfilerStop()

print("Profiling complete - both square and sin operations")