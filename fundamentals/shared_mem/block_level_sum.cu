#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <random>

#include "functions.h"

__global__ void sum(float *A, float *B, float *C, int N)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        sdata[tid] = A[i] + B[i];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        C[blockIdx.x] = sdata[0];
}

int main()
{
    int N = 1024;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));

    initialize_random_vals(h_A, N);
    initialize_random_vals(h_B, N);
    print_array_vals(h_A, N);
    print_array_vals(h_B, N);

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, blocks * sizeof(float));
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    sum<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    print_array_vals(h_C, blocks);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}