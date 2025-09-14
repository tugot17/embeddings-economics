#!/usr/bin/env python3
"""
Compare H100 vs 4090 embedding performance across different sequence lengths and batch sizes
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

def load_embedding_data(results_dir, gpu_type, word_count_filter=None):
    """Load all embedding result files and extract performance data"""
    data = {}
    
    # Get all JSON files in the directory
    json_files = list(Path(results_dir).glob(f"embedding_results_*_{gpu_type}.json"))
    
    for file_path in sorted(json_files):
        with open(file_path, 'r') as f:
            result = json.load(f)
        
        # Extract sequence length from metadata
        seq_length = result['metadata']['sequence_lengths'][0]
        
        # Apply word count filter if specified
        if word_count_filter and not re.match(word_count_filter, str(seq_length)):
            continue
            
        batch_sizes = result['metadata']['batch_sizes']
        
        # Extract performance data for each batch size
        batch_data = []
        embeddings_per_sec = []
        embeddings_std = []
        actual_tokens = None
        
        for batch_size in batch_sizes:
            key = f"batch_{batch_size}_seq_{seq_length}"
            if key in result['results']:
                batch_result = result['results'][key]
                
                # Get actual tokens per batch (this is the real token count)
                tokens_per_batch = batch_result['tokens']['total_per_batch']['mean']
                if actual_tokens is None:
                    actual_tokens = int(tokens_per_batch)
                
                # Get embeddings per second with std
                eps_mean = batch_result['throughput']['batch_embeddings_per_second']['mean']
                eps_std = batch_result['throughput']['batch_embeddings_per_second']['std']
                
                batch_data.append(batch_size)
                embeddings_per_sec.append(eps_mean)
                embeddings_std.append(eps_std)
        
        # Use actual token count as the key, not sequence length
        if batch_data:  # Only add if we have data
            data[actual_tokens] = {
                'batch_sizes': batch_data,
                'embeddings_per_sec': embeddings_per_sec,
                'embeddings_std': embeddings_std,
                'tokens_per_seq': actual_tokens,
                'words': seq_length  # Keep track of word count too
            }
    
    return data

def plot_comparison_performance(h100_data, rtx4090_data, word_count_filter_desc=""):
    """Create comparison performance plot"""
    
    plt.figure(figsize=(14, 10))
    
    # Get all unique token counts from both datasets
    all_token_counts = sorted(set(list(h100_data.keys()) + list(rtx4090_data.keys())))
    
    # Colors for different sequence lengths (same color for same token count across GPUs)
    # Use colors similar to your reference image - purple, blue, teal, green (avoiding harsh yellow)
    if len(all_token_counts) <= 5:
        custom_colors = ['#440154', '#31688e', '#35b779', '#1f9e89', '#26828e']  # Purple to blue-green gradient
        colors = [custom_colors[i] for i in range(len(all_token_counts))]
    else:
        # For more colors, use viridis but exclude the yellow end
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(all_token_counts)))  # Stop at 0.85 to avoid yellow
    color_map = {token_count: colors[i] for i, token_count in enumerate(all_token_counts)}
    
    # Plot H100 data (solid lines)
    for token_count, seq_data in sorted(h100_data.items()):
        batch_sizes = seq_data['batch_sizes']
        embeddings_per_sec = seq_data['embeddings_per_sec']
        embeddings_std = seq_data['embeddings_std']
        
        color = color_map[token_count]
        
        # Plot solid line for H100
        plt.plot(batch_sizes, embeddings_per_sec, marker='o', linewidth=2, 
                markersize=6, label=f'H100 - {token_count} tokens', 
                color=color, linestyle='-')
        
        # Fill between for std deviation area
        plt.fill_between(batch_sizes, 
                        np.array(embeddings_per_sec) - np.array(embeddings_std),
                        np.array(embeddings_per_sec) + np.array(embeddings_std),
                        alpha=0.15, color=color)
        
        # Add value labels above each dot
        for j, (batch_size, eps_mean, eps_std) in enumerate(zip(batch_sizes, embeddings_per_sec, embeddings_std)):
            plt.annotate(f'{eps_mean:.1f}', 
                        (batch_size, eps_mean),
                        textcoords="offset points", 
                        xytext=(0,12), 
                        ha='center', 
                        fontsize=7,
                        color=color,
                        fontweight='bold')
    
    # Plot 4090 data (dotted lines)
    for token_count, seq_data in sorted(rtx4090_data.items()):
        batch_sizes = seq_data['batch_sizes']
        embeddings_per_sec = seq_data['embeddings_per_sec']
        embeddings_std = seq_data['embeddings_std']
        
        color = color_map[token_count]
        
        # Plot dotted line for 4090
        plt.plot(batch_sizes, embeddings_per_sec, marker='s', linewidth=2, 
                markersize=6, label=f'RTX 4090 - {token_count} tokens', 
                color=color, linestyle='--')
        
        # Fill between for std deviation area
        plt.fill_between(batch_sizes, 
                        np.array(embeddings_per_sec) - np.array(embeddings_std),
                        np.array(embeddings_per_sec) + np.array(embeddings_std),
                        alpha=0.15, color=color)
        
        # Add value labels below each dot (offset down to avoid overlap)
        for j, (batch_size, eps_mean, eps_std) in enumerate(zip(batch_sizes, embeddings_per_sec, embeddings_std)):
            plt.annotate(f'{eps_mean:.1f}', 
                        (batch_size, eps_mean),
                        textcoords="offset points", 
                        xytext=(0,-15), 
                        ha='center', 
                        fontsize=7,
                        color=color,
                        fontweight='bold')
    
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Embeddings/sec', fontsize=12)
    
    title = f'H100 vs RTX 4090 Embedding Performance Comparison\nQwen3-8B Embeddings/sec vs Batch Size'
    if word_count_filter_desc:
        title += f' ({word_count_filter_desc})'
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Set x-axis to log scale to match the reference
    if all_token_counts and h100_data:
        sample_batch_sizes = h100_data[list(h100_data.keys())[0]]['batch_sizes']
        plt.xscale('log', base=2)
        plt.xticks(sorted(sample_batch_sizes))
        plt.gca().set_xticklabels([str(x) for x in sorted(sample_batch_sizes)])
    elif all_token_counts and rtx4090_data:
        sample_batch_sizes = rtx4090_data[list(rtx4090_data.keys())[0]]['batch_sizes']
        plt.xscale('log', base=2)
        plt.xticks(sorted(sample_batch_sizes))
        plt.gca().set_xticklabels([str(x) for x in sorted(sample_batch_sizes)])
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend with custom ordering (H100 first, then 4090 for each token count)
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Reorder legend: group by token count, H100 first then 4090
    ordered_handles = []
    ordered_labels = []
    
    for token_count in all_token_counts:
        # Add H100 entry first
        for i, label in enumerate(labels):
            if f'H100 - {token_count} tokens' == label:
                ordered_handles.append(handles[i])
                ordered_labels.append(label)
                break
        
        # Add 4090 entry second
        for i, label in enumerate(labels):
            if f'RTX 4090 - {token_count} tokens' == label:
                ordered_handles.append(handles[i])
                ordered_labels.append(label)
                break
    
    plt.legend(ordered_handles, ordered_labels, loc='upper left', fontsize=9, ncol=2)
    
    # Tight layout
    plt.tight_layout()
    
    return plt

def main():
    # Configuration - specify which word counts to include
    word_count_pattern = r'^(2000|4000)$'  # Regex pattern for word counts to include
    word_count_desc = "2000 and 4000 words"  # Description for the title

    # word_count_pattern = r'^(400|800)$'  # Regex pattern for word counts to include
    # word_count_desc = "400 and 800 words"  # Description for the title
    
    # Directories containing the embedding results (relative to script location)
    script_dir = Path(__file__).parent
    h100_results_dir = script_dir / "h100_embedding_results"
    rtx4090_results_dir = script_dir / "4090_embedding_results"
    
    # Load data for both GPUs
    print("Loading H100 embedding performance data...")
    h100_data = load_embedding_data(h100_results_dir, "h100", word_count_pattern)
    print(f"H100 - Found data for token counts: {sorted(h100_data.keys())}")
    
    print("Loading RTX 4090 embedding performance data...")
    rtx4090_data = load_embedding_data(rtx4090_results_dir, "4090", word_count_pattern)
    print(f"RTX 4090 - Found data for token counts: {sorted(rtx4090_data.keys())}")
    
    if not h100_data and not rtx4090_data:
        print("No embedding data found for either GPU!")
        print(f"Make sure the directories exist and contain embedding result files matching pattern: {word_count_pattern}")
        return
    
    if not h100_data:
        print(f"No H100 data found in {h100_results_dir}")
    
    if not rtx4090_data:
        print(f"No RTX 4090 data found in {rtx4090_results_dir}")
    
    # Create comparison plot
    print("Creating comparison plot...")
    plt = plot_comparison_performance(h100_data, rtx4090_data, word_count_desc)
    
    # Save plot (relative to script location)
    output_path = script_dir / "h100_vs_4090_embedding_performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
