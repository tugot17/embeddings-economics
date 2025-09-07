import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_benchmark_data(file_4090, file_h100):
    """Load and parse the benchmark JSON files"""
    with open(file_4090, 'r') as f:
        data_4090 = json.load(f)
    
    with open(file_h100, 'r') as f:
        data_h100 = json.load(f)
    
    return data_4090, data_h100

def extract_performance_data(data_4090, data_h100):
    """Extract embeddings per second for each batch size"""
    batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512]
    
    performance_data = {
        'batch_size': batch_sizes,
        'rtx_4090_mean': [],
        'rtx_4090_std': [],
        'h100_mean': [],
        'h100_std': []
    }
    
    for batch in batch_sizes:
        key = f"batch_{batch}_seq_2000"
        
        rtx_mean = data_4090['results'][key]['throughput']['batch_embeddings_per_second']['mean']
        rtx_std = data_4090['results'][key]['throughput']['batch_embeddings_per_second']['std']
        h100_mean = data_h100['results'][key]['throughput']['batch_embeddings_per_second']['mean']
        h100_std = data_h100['results'][key]['throughput']['batch_embeddings_per_second']['std']
        
        performance_data['rtx_4090_mean'].append(rtx_mean)
        performance_data['rtx_4090_std'].append(rtx_std)
        performance_data['h100_mean'].append(h100_mean)
        performance_data['h100_std'].append(h100_std)
    
    return pd.DataFrame(performance_data)

def create_comparison_plot(df, title="GPU Embedding Performance Comparison", seq_length=256):
    """Create a single clean comparison plot"""
    
    # Set up plotting style to match the reference
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors matching the reference style
    color_4090 = '#1f77b4'  # Blue
    color_h100 = '#2ca02c'  # Green
    
    # Plot RTX 4090 line with confidence interval
    ax.plot(df['batch_size'], df['rtx_4090_mean'], 
            color=color_4090, linewidth=3, marker='o', 
            markersize=8, label='RTX 4090')
    ax.fill_between(df['batch_size'],
                    df['rtx_4090_mean'] - df['rtx_4090_std'],
                    df['rtx_4090_mean'] + df['rtx_4090_std'],
                    color=color_4090, alpha=0.2)
    
    # Plot H100 line with confidence interval
    ax.plot(df['batch_size'], df['h100_mean'], 
            color=color_h100, linewidth=3, marker='s', 
            markersize=8, label='H100')
    ax.fill_between(df['batch_size'],
                    df['h100_mean'] - df['h100_std'],
                    df['h100_mean'] + df['h100_std'],
                    color=color_h100, alpha=0.2)
    
    # Add value annotations for key points
    key_indices = [0, 3, 6, 7]  # batch sizes 1, 32, 256, 512
    for i in key_indices:
        # RTX 4090 annotations
        ax.annotate(f'{df.iloc[i]["rtx_4090_mean"]:.1f} ± {df.iloc[i]["rtx_4090_std"]:.1f}',
                   xy=(df.iloc[i]['batch_size'], df.iloc[i]['rtx_4090_mean']), 
                   xytext=(0, 20), textcoords='offset points', ha='center',
                   fontsize=10, fontweight='bold', color=color_4090)
        
        # H100 annotations
        ax.annotate(f'{df.iloc[i]["h100_mean"]:.1f} ± {df.iloc[i]["h100_std"]:.1f}',
                   xy=(df.iloc[i]['batch_size'], df.iloc[i]['h100_mean']), 
                   xytext=(0, -25), textcoords='offset points', ha='center',
                   fontsize=10, fontweight='bold', color=color_h100)
    
    # Styling
    ax.set_title(f'{title}\nQwen3-Embedding-8B - {seq_length} Tokens', 
                pad=20, fontsize=16, fontweight='bold')
    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Embeddings/sec', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='center right')
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set log scale for x-axis and format ticks
    ax.set_xscale('log', base=2)
    ax.set_xticks(df['batch_size'])
    ax.set_xticklabels(df['batch_size'])
    
    # Add performance summary text box in upper left
    max_4090 = df['rtx_4090_mean'].max()
    max_h100 = df['h100_mean'].max()
    max_speedup = (df['h100_mean'] / df['rtx_4090_mean']).max()
    
    textstr = f'Max Speedup: {max_speedup:.1f}x'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main(file_4090='embedding_results_200_4900.json', 
         file_h100='embedding_results_200_h100.json',
         title="GPU Embedding Performance Comparison",
         seq_length=256):
    """Main function to create and save the comparison plot"""
    
    try:
        # Load and process data
        data_4090, data_h100 = load_benchmark_data(file_4090, file_h100)
        df = extract_performance_data(data_4090, data_h100)
        
        # Create plot
        fig = create_comparison_plot(df, title, seq_length)
        
        # Save plot
        output_filename = 'gpu_embedding_comparison.png'
        fig.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Plot saved as: {output_filename}")
        
        # Print summary
        print("\nPerformance Summary:")
        print(f"RTX 4090 Peak: {df['rtx_4090_mean'].max():.1f} embeddings/sec")
        print(f"H100 Peak: {df['h100_mean'].max():.1f} embeddings/sec")
        speedup = df['h100_mean'] / df['rtx_4090_mean']
        print(f"Average Speedup: {speedup.mean():.1f}x")
        print(f"Max Speedup: {speedup.max():.1f}x")
        
        return df
        
    except FileNotFoundError as e:
        print(f"Error: Could not find benchmark files. Please check file paths.")
        print(f"Expected files: {file_4090}, {file_h100}")
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    # Example usage with custom title and sequence length
    main(
        file_4090='embedding_results_2000_4900.json',
        file_h100='embedding_results_2000_h100.json', 
        title="RTX 4090 vs H100 Performance",
        seq_length=2560  # Change this to match your actual token length
    )