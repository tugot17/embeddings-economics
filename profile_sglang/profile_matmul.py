"""
Simple FLOPS measurement script for H100 with BF16 matrix multiplication.
Baseline: 989 TFLOPs for H100
"""

import torch
import triton
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def measure_matmul_flops(M: int, N: int, K: int, dtype=torch.bfloat16, warmup: int = 25, rep: int = 100) -> float:
    """
    Measure FLOPS for matrix multiplication A @ B where A is (M, K) and B is (K, N).
    
    Args:
        M, N, K: Matrix dimensions
        dtype: Data type (default: bfloat16)
        warmup: Number of warmup iterations
        rep: Number of measurement repetitions
    
    Returns:
        FLOPS in TFLOPs/s
    """
    device = torch.cuda.current_device()
    
    # Create random matrices on GPU
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    
    # Define the operation to benchmark
    def matmul_op():
        return torch.matmul(A, B)
    
    # Measure time using Triton's do_bench
    time_ms = triton.testing.do_bench(matmul_op, warmup=warmup, rep=rep)
    
    # Calculate FLOPS
    # For matrix multiplication: 2 * M * N * K operations (multiply + add for each element)
    total_ops = 2 * M * N * K
    flops_per_second = total_ops / (time_ms * 1e-3)  # Convert ms to seconds
    tflops_per_second = flops_per_second / 1e12  # Convert to TFLOPs
    
    return tflops_per_second

def run_flops_benchmark(matrix_sizes: List[int] = None) -> Tuple[List[int], List[float]]:
    """
    Run FLOPS benchmark for different matrix sizes.
    
    Args:
        matrix_sizes: List of matrix sizes to test (default: [512, 1024, 2048, 4096, 8192, 16384])
    
    Returns:
        Tuple of (matrix_sizes, tflops_results)
    """
    if matrix_sizes is None:
        matrix_sizes = [512, 1024, 2048, 4096, 8192, 16384]
    
    print("Starting BF16 FLOPS benchmark...")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"H100 Baseline target: 989 TFLOPs")
    print("-" * 60)
    
    tflops_results = []
    
    for size in matrix_sizes:
        print(f"Testing {size}x{size} matrices...", end=" ")
        
        tflops = measure_matmul_flops(size, size, size)
        tflops_results.append(tflops)
        
        print(f"{tflops:.2f} TFLOPs")
    
    print("-" * 60)
    print(f"Peak performance: {max(tflops_results):.2f} TFLOPs")
    print(f"H100 baseline ratio: {max(tflops_results)/989:.2%}")
    
    return matrix_sizes, tflops_results


def plot_results(matrix_sizes: List[int], tflops_results: List[float], save_path: str = "h100_bf16_benchmark.png"):
    """
    Plot FLOPS performance vs matrix size and save to file.
    
    Args:
        matrix_sizes: List of matrix sizes tested
        tflops_results: Corresponding FLOPS measurements
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the results
    plt.plot(matrix_sizes, tflops_results, 'bo-', linewidth=2, markersize=8, label='Measured FLOPS')
    
    # Add H100 baseline reference line
    plt.axhline(y=989, color='r', linestyle='--', linewidth=2, label='H100 Baseline (989 TFLOPs)')
    
    # Formatting
    plt.xlabel('Matrix Size (NxN)', fontsize=12)
    plt.ylabel('Performance (TFLOPs)', fontsize=12)
    plt.title('H100 BF16 Matrix Multiplication FLOPS Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Use log scale for x-axis to better show the range
    plt.xscale('log')
    plt.xticks(matrix_sizes, [str(s) for s in matrix_sizes])
    
    # Add performance annotations
    peak_idx = np.argmax(tflops_results)
    peak_size = matrix_sizes[peak_idx]
    peak_flops = tflops_results[peak_idx]
    plt.annotate(f'Peak: {peak_flops:.1f} TFLOPs\n@{peak_size}x{peak_size}', 
                xy=(peak_size, peak_flops), 
                xytext=(peak_size*0.5, peak_flops*1.1),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.show()


def main():
    """Main function to run the H100 BF16 FLOPS benchmark."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")
    
    print("=" * 60)
    print("H100 BF16 Matrix Multiplication FLOPS Benchmark")
    print("=" * 60)
    
    # Run the benchmark
    matrix_sizes, tflops_results = run_flops_benchmark()
    
    # Plot results
    plot_results(matrix_sizes, tflops_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Peak performance: {max(tflops_results):.2f} TFLOPs")
    print(f"H100 baseline ratio: {max(tflops_results)/989:.1%}")
    
    if max(tflops_results) > 989:
        print(f"\n🎉 SUCCESS: Achieved {max(tflops_results):.2f} TFLOPs, exceeding the 989 TFLOPs H100 baseline!")
    else:
        print(f"\n📊 Result: Achieved {max(tflops_results):.2f} TFLOPs ({max(tflops_results)/989:.1%} of H100 baseline)")

if __name__ == "__main__":
    main()