#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Simple vector addition kernel - good for basic profiling
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Matrix multiplication kernel - more complex for advanced profiling
__global__ void matrixMul(float *a, float *b, float *c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = sum;
    }
}

int main() {
    // Vector addition example
    const int vectorSize = 1024 * 1024;  // 1M elements
    const size_t vectorBytes = vectorSize * sizeof(float);
    
    // Host memory allocation
    float *h_a = (float*)malloc(vectorBytes);
    float *h_b = (float*)malloc(vectorBytes);
    float *h_c = (float*)malloc(vectorBytes);
    
    // Initialize vectors
    for (int i = 0; i < vectorSize; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // Device memory allocation
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, vectorBytes);
    cudaMalloc(&d_b, vectorBytes);
    cudaMalloc(&d_c, vectorBytes);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vectorBytes, cudaMemcpyHostToDevice);
    
    // Launch vector addition kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching vector addition: %d blocks, %d threads per block\n", 
           blocksPerGrid, threadsPerBlock);
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, vectorSize);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_c, d_c, vectorBytes, cudaMemcpyDeviceToHost);
    
    // Verify first few results
    printf("Vector addition results (first 5): ");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", h_c[i]);
    }
    printf("\n");
    
    // Matrix multiplication example
    const int matrixWidth = 512;
    const size_t matrixBytes = matrixWidth * matrixWidth * sizeof(float);
    
    // Host matrices
    float *h_ma = (float*)malloc(matrixBytes);
    float *h_mb = (float*)malloc(matrixBytes);
    float *h_mc = (float*)malloc(matrixBytes);
    
    // Initialize matrices
    for (int i = 0; i < matrixWidth * matrixWidth; i++) {
        h_ma[i] = 1.0f;
        h_mb[i] = 2.0f;
    }
    
    // Device matrices
    float *d_ma, *d_mb, *d_mc;
    cudaMalloc(&d_ma, matrixBytes);
    cudaMalloc(&d_mb, matrixBytes);
    cudaMalloc(&d_mc, matrixBytes);
    
    // Copy matrices to device
    cudaMemcpy(d_ma, h_ma, matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mb, h_mb, matrixBytes, cudaMemcpyHostToDevice);
    
    // Launch matrix multiplication
    dim3 blockDim(16, 16);
    dim3 gridDim((matrixWidth + blockDim.x - 1) / blockDim.x,
                 (matrixWidth + blockDim.y - 1) / blockDim.y);
    
    printf("Launching matrix multiplication: (%d,%d) grid, (%d,%d) blocks\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    matrixMul<<<gridDim, blockDim>>>(d_ma, d_mb, d_mc, matrixWidth);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_mc, d_mc, matrixBytes, cudaMemcpyDeviceToHost);
    
    // Verify result (should be matrixWidth * 2.0 for each element)
    printf("Matrix multiplication result [0][0]: %.1f (expected: %.1f)\n", 
           h_mc[0], (float)matrixWidth * 2.0f);
    
    // Cleanup
    free(h_a); free(h_b); free(h_c);
    free(h_ma); free(h_mb); free(h_mc);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_ma); cudaFree(d_mb); cudaFree(d_mc);
    
    printf("Profiling example completed successfully!\n");
    return 0;
}