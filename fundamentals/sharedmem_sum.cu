#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <random>

#include "functions.h"

#define TILE 16

__global__ void sum(float *A, float *B, int size)
{
    __shared__ float sdata[TILE * TILE];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int global_idx = blockIdx.y * blockDim.y * gridDim.x * blockDim.x +
                     blockIdx.x * blockDim.x +
                     tid;

    if (global_idx < size)
        sdata[tid] = A[global_idx];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        int block_id = blockIdx.y * gridDim.x + blockIdx.x;
        B[block_id] = sdata[0];
    }
}

int main()
{
    int N = 32;
    int size = N * N;
    dim3 threads(16, 16);
    dim3 blocks((N - 1) / threads.x + 1, (N - 1) / threads.y + 1);

    float *h_A = (float *)malloc(size * sizeof(float));
    float *h_B = (float *)malloc(size * sizeof(float));

    initialize_random_vals(h_A, size);
    print_array_vals(h_A, size);

    float *d_A;
    float *d_B;

    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);

    sum<<<blocks, threads>>>(d_A, d_B, N);

    cudaMemcpy(h_A, d_A, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}