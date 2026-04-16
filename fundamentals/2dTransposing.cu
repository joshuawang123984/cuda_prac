#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <random>

#include "functions.h"

__global__ void transpose(float *A, float *B, int N)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    for (int r = row; r < N; r += blockDim.y * gridDim.y)
    {
        for (int c = col; c < N; c += blockDim.x * gridDim.x)
        {
            B[r * N + c] = A[c * N + r];
        }
    }
}

int main()
{
    int N = 64;
    int size = N * N;
    dim3 threads(8, 8);
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

    transpose<<<blocks, threads>>>(d_A, d_B, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);

    print_array_vals(h_B, size);

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}