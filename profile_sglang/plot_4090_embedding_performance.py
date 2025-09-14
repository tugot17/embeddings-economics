#!/usr/bin/env python3
"""
Plot 4090 embedding performance across different sequence lengths and batch sizes
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_embedding_data(results_dir):
    """Load all embedding result files and extract performance data"""
    data = {}
    
    # Get all JSON files in the directory
    json_files = list(Path(results_dir).glob("embedding_results_*_4090.json"))
    
    for file_path in sorted(json_files):
        with open(file_path, 'r') as f:
            result = json.load(f)
        
        # Extract sequence length from metadata
        seq_length = result['metadata']['sequence_lengths'][0]
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
        data[actual_tokens] = {
            'batch_sizes': batch_data,
            'embeddings_per_sec': embeddings_per_sec,
            'embeddings_std': embeddings_std,
            'tokens_per_seq': actual_tokens,
            'words': seq_length  # Keep track of word count too
        }
    
    return data

def plot_embedding_performance(data):
    """Create performance plot similar to the reference image"""
    
    # Description from the JSON metadata
    description = "RTX 4090 TP1 Qwen3-8B Embedding Performance"
    
    plt.figure(figsize=(12, 8))
    
    # Colors for different sequence lengths
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    
    for i, (token_count, seq_data) in enumerate(sorted(data.items())):
        batch_sizes = seq_data['batch_sizes']
        embeddings_per_sec = seq_data['embeddings_per_sec']
        embeddings_std = seq_data['embeddings_std']
        
        # Plot line without error bars
        plt.plot(batch_sizes, embeddings_per_sec, marker='o', linewidth=2, 
                markersize=6, label=f'{token_count} tokens', color=colors[i])
        
        # Fill between for std deviation area
        plt.fill_between(batch_sizes, 
                        np.array(embeddings_per_sec) - np.array(embeddings_std),
                        np.array(embeddings_per_sec) + np.array(embeddings_std),
                        alpha=0.2, color=colors[i])
        
        # Add value labels above each dot
        for j, (batch_size, eps_mean, eps_std) in enumerate(zip(batch_sizes, embeddings_per_sec, embeddings_std)):
            plt.annotate(f'{eps_mean:.1f} ± {eps_std:.1f}', 
                        (batch_size, eps_mean),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center', 
                        fontsize=8,
                        color=colors[i],
                        fontweight='bold')
    
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Embeddings/sec', fontsize=12)
    plt.title(f'{description}\nEmbeddings/sec vs Batch Size', fontsize=14, fontweight='bold')
    
    # Set x-axis to log scale to match the reference
    plt.xscale('log', base=2)
    plt.xticks(sorted(data[list(data.keys())[0]]['batch_sizes']))
    plt.gca().set_xticklabels([str(x) for x in sorted(data[list(data.keys())[0]]['batch_sizes'])])
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(loc='upper left', fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    return plt

def main():
    # Directory containing the embedding results (relative to script location)
    script_dir = Path(__file__).parent
    results_dir = script_dir / "4090_embedding_results_v2"
    
    # Load data
    print("Loading embedding performance data...")
    data = load_embedding_data(results_dir)
    
    print(f"Found data for token counts: {sorted(data.keys())}")
    
    if not data:
        print(f"No embedding data found in {results_dir}")
        print("Make sure the directory exists and contains embedding_results_*_4090.json files")
        return
    
    # Create plot
    print("Creating performance plot...")
    plt = plot_embedding_performance(data)
    
    # Save plot (relative to script location)
    output_path = script_dir / "4090_embedding_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
