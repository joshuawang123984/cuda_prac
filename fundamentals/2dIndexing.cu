#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <random>

#include "functions.h"

__global__ void matrix_add(float *A, float *B, float *C, int N)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    for (int r = row; r < N; r += blockDim.y * gridDim.y)
    {
        for (int c = col; c < N; c += blockDim.x * gridDim.x)
        {
            int idx = r * N + c;
            C[idx] = A[idx] + B[idx];
        }
    }
}

int main()
{
    int N = 256;
    int size = N * N;
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);

    float *h_A = (float *)malloc(size * sizeof(float));
    float *h_B = (float *)malloc(size * sizeof(float));
    float *h_C = (float *)malloc(size * sizeof(float));

    initialize_random_vals(h_A, size);
    initialize_random_vals(h_B, size);
    print_array_vals(h_A, size);
    print_array_vals(h_B, size);

    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

    matrix_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
    print_array_vals(h_C, size);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}