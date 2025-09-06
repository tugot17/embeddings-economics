#!/usr/bin/env python3
import json
import gzip
import sys

def fix_prepare_varlen_duration(input_file, output_file):
    print(f"Processing: {input_file}")
    
    # Load trace
    with gzip.open(input_file, 'rt') as f:
        data = json.load(f)
    
    # Fix the problematic kernel durations
    events = data.get('traceEvents', data)
    fixed_count = 0
    
    for event in events:
        name = event.get('name', '')
        if 'flash::prepare_varlen_num_blocks_kernel' in name:
            event['dur'] = 1.0  # Set duration to 1 microsecond
            fixed_count += 1
    
    print(f"Fixed {fixed_count} prepare_varlen_num_blocks_kernel events")
    
    # Save fixed trace
    with gzip.open(output_file, 'wt') as f:
        json.dump(data, f)
    
    print(f"Saved fixed trace to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python quick_fix_trace.py input.gz output.gz")
        sys.exit(1)
    
    fix_prepare_varlen_duration(sys.argv[1], sys.argv[2])