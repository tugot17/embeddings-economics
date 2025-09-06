#!/usr/bin/env python3
import json
import gzip
from collections import defaultdict
import sys

def analyze_trace(trace_file):
    print(f"Analyzing trace file: {trace_file}")
    
    kernel_durations = defaultdict(float)
    kernel_counts = defaultdict(int)
    total_events = 0
    
    # Handle both .gz and regular files
    if trace_file.endswith('.gz'):
        with gzip.open(trace_file, 'rt') as f:
            data = json.load(f)
    else:
        with open(trace_file, 'r') as f:
            data = json.load(f)
    
    # Process trace events
    if 'traceEvents' in data:
        events = data['traceEvents']
    else:
        events = data
    
    for event in events:
        total_events += 1
            
        # Look for GPU kernel events
        if event.get('cat') == 'kernel' or 'kernel' in event.get('name', '').lower():
            name = event.get('name', 'unknown')
            duration = event.get('dur', 0)  # Duration in microseconds
            kernel_durations[name] += duration
            kernel_counts[name] += 1
        
        # Also check for CUDA events
        elif 'cuda' in event.get('name', '').lower() or event.get('cat') == 'cuda_runtime':
            name = event.get('name', 'unknown')
            duration = event.get('dur', 0)
            kernel_durations[name] += duration
            kernel_counts[name] += 1
    
    print(f"\nTotal events processed: {total_events}")
    print(f"Found {len(kernel_durations)} unique kernel types")
    
    # Sort by total duration (descending)
    sorted_kernels = sorted(kernel_durations.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 kernels by total duration:")
    print("=" * 100)
    print(f"{'Kernel Name':<40} {'Total Duration (ms)':<18} {'Count':<8} {'Avg Duration (ms)':<18} {'Percentage':<10}")
    print("=" * 100)
    
    total_duration = sum(kernel_durations.values())
    
    for i, (kernel_name, duration_us) in enumerate(sorted_kernels[:20]):
        duration_ms = duration_us / 1000.0  # Convert to milliseconds
        count = kernel_counts[kernel_name]
        avg_duration_ms = duration_ms / count if count > 0 else 0
        percentage = (duration_us / total_duration * 100) if total_duration > 0 else 0
        print(f"{kernel_name[:39]:<40} {duration_ms:<18.2f} {count:<8} {avg_duration_ms:<18.2f} {percentage:<10.1f}%")
    
    print("=" * 80)
    print(f"Total kernel time: {total_duration / 1000.0:.2f} ms")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analzye_trace.py <trace_file>")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    analyze_trace(trace_file)
