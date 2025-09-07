#!/usr/bin/env python3
import json
import gzip
from collections import defaultdict
import sys

def analyze_trace(trace_file):
    print(f"Analyzing trace file: {trace_file}")
    
    kernel_durations = defaultdict(float)
    kernel_counts = defaultdict(int)
    memcpy_durations = defaultdict(float)
    memcpy_counts = defaultdict(int)
    api_durations = defaultdict(float)
    api_counts = defaultdict(int)
    
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
    
    # Group events by category for better analysis
    for event in events:
        total_events += 1
        name = event.get('name', 'unknown')
        duration = event.get('dur', 0)  # Duration in microseconds
        category = event.get('cat', '')
        
        # Filter for actual GPU kernel execution (not API calls)
        if category == 'kernel':
            kernel_durations[name] += duration
            kernel_counts[name] += 1
            
        # Separate memory copy operations
        elif 'memcpy' in name.lower() or 'Memcpy' in name:
            memcpy_durations[name] += duration
            memcpy_counts[name] += 1
            
        # CUDA API calls (often inflated due to synchronization)
        elif category in ['cuda_runtime', 'cuda_api']:
            api_durations[name] += duration
            api_counts[name] += 1
            
        # Fallback: classify based on name patterns
        elif any(pattern in name.lower() for pattern in ['kernel', 'grid', 'block']):
            kernel_durations[name] += duration
            kernel_counts[name] += 1
    
    print(f"\nTotal events processed: {total_events}")
    print(f"Found {len(kernel_durations)} unique kernel types")
    print(f"Found {len(memcpy_durations)} memory copy operations")
    print(f"Found {len(api_durations)} API calls")
    
    # Method 1: Show kernels only (excluding API overhead)
    print("\n" + "="*100)
    print("COMPUTE KERNELS ONLY (excluding API overhead)")
    print("="*100)
    print_kernel_analysis(kernel_durations, kernel_counts, "Compute Kernels")
    
    # Method 2: Show all categories separately
    print("\n" + "="*100)
    print("MEMORY OPERATIONS")
    print("="*100)
    print_kernel_analysis(memcpy_durations, memcpy_counts, "Memory Operations")
    
    print("\n" + "="*100)
    print("CUDA API CALLS (may include synchronization overhead)")
    print("="*100)
    print_kernel_analysis(api_durations, api_counts, "API Calls")
    
    # Method 3: Timeline-based analysis (if timestamps available)
    print("\n" + "="*100)
    print("TIMELINE ANALYSIS")
    print("="*100)
    analyze_timeline_overlaps(events)

def print_kernel_analysis(durations_dict, counts_dict, category_name):
    if not durations_dict:
        print(f"No {category_name} found")
        return
        
    sorted_items = sorted(durations_dict.items(), key=lambda x: x[1], reverse=True)
    total_duration = sum(durations_dict.values())
    
    print(f"{'Name':<40} {'Total (ms)':<12} {'Count':<8} {'Avg (ms)':<12} {'%':<8}")
    print("-" * 85)
    
    for i, (name, duration_us) in enumerate(sorted_items[:20]):
        duration_ms = duration_us / 1000.0
        count = counts_dict[name]
        avg_duration_ms = duration_ms / count if count > 0 else 0
        percentage = (duration_us / total_duration * 100) if total_duration > 0 else 0
        print(f"{name[:39]:<40} {duration_ms:<12.2f} {count:<8} {avg_duration_ms:<12.2f} {percentage:<8.1f}%")
    
    print(f"\nTotal {category_name} time: {total_duration / 1000.0:.2f} ms")

def analyze_timeline_overlaps(events):
    """Analyze timeline to detect overlapping operations"""
    # Filter events with timestamps
    timestamped_events = []
    for event in events:
        if 'ts' in event and 'dur' in event:
            start_time = event['ts']
            end_time = start_time + event['dur']
            timestamped_events.append({
                'name': event.get('name', 'unknown'),
                'category': event.get('cat', ''),
                'start': start_time,
                'end': end_time,
                'duration': event['dur']
            })
    
    if not timestamped_events:
        print("No timestamp data available for overlap analysis")
        return
    
    # Sort by start time
    timestamped_events.sort(key=lambda x: x['start'])
    
    # Find overlapping memcpy and kernel operations
    overlaps = []
    memcpy_events = [e for e in timestamped_events if 'memcpy' in e['name'].lower()]
    kernel_events = [e for e in timestamped_events if e['category'] == 'kernel']
    
    overlap_time = 0
    for memcpy in memcpy_events:
        for kernel in kernel_events:
            # Check if they overlap
            overlap_start = max(memcpy['start'], kernel['start'])
            overlap_end = min(memcpy['end'], kernel['end'])
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                overlap_time += overlap_duration
                overlaps.append({
                    'memcpy': memcpy['name'],
                    'kernel': kernel['name'], 
                    'overlap_duration': overlap_duration
                })
    
    if overlaps:
        print(f"Found {len(overlaps)} overlapping operations")
        print(f"Total overlap time: {overlap_time / 1000.0:.2f} ms")
        print("This suggests memory copies are running concurrently with kernels")
    else:
        print("No significant overlaps detected - operations appear sequential")

def advanced_analysis(trace_file):
    """Alternative analysis focusing on wall-clock time"""
    print(f"\n{'='*100}")
    print("ADVANCED: WALL-CLOCK TIME ANALYSIS")
    print(f"{'='*100}")
    
    # This would require more sophisticated timeline analysis
    # to calculate actual wall-clock execution time accounting for parallelism
    print("For true wall-clock analysis, consider using:")
    print("1. nsys stats <trace_file> --report kernel")
    print("2. ncu --metrics gpu__time_duration.sum <your_program>")
    print("3. Or use this script with --wall-clock flag (implementation needed)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_trace.py <trace_file> [--advanced]")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    
    if len(sys.argv) > 2 and sys.argv[2] == '--advanced':
        advanced_analysis(trace_file)
    
    analyze_trace(trace_file)