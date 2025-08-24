import torch
import gzip
import json
from collections import defaultdict

# Option 1: Parse the trace JSON directly (your trace file)
with gzip.open('./traces/hysterical-millipede-of-flowers_32019.1756053484681442417.pt.trace.json.gz', 'rt') as f:
    trace_data = json.load(f)

# Extract CUDA kernel events and sum by name
kernel_times = defaultdict(float)
for event in trace_data.get('traceEvents', []):
    if event.get('cat') == 'kernel' and 'dur' in event:
        kernel_name = event['name']
        duration = event['dur']  # microseconds
        kernel_times[kernel_name] += duration

# Sort and display top kernels
sorted_kernels = sorted(kernel_times.items(), key=lambda x: x[1], reverse=True)
print("Top 10 CUDA kernels by GPU time:")
for kernel, time_us in sorted_kernels[:10]:
    print(f"{time_us:>10.2f} μs  {kernel}")