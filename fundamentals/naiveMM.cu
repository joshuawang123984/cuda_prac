#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <random>

#include "functions.h"

__global__ void matrix_multiply(float *A, float *B, float *C, int N, int K, int M)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < M)
    {
        sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[M * k + col];
        }

        C[row * M + col] = sum;
    }
}

int main()
{
    int N = 32;
    int K = 32;
    int M = 32;

    int A_size = N * K;
    int B_size = K * M;
    int C_size = N * M;

    dim3 threads(16, 16);
    dim3 blocks((M - 1) / threads.x + 1, (N - 1) / threads.y + 1);

    float *h_A = (float *)malloc(A_size * sizeof(float));
    float *h_B = (float *)malloc(B_size * sizeof(float));
    float *h_C = (float *)malloc(C_size * sizeof(float));

    initialize_random_vals(h_A, A_size);
    initialize_random_vals(h_B, B_size);
    print_array_vals(h_A, A_size);
    print_array_vals(h_B, B_size);

    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc(&d_A, A_size * sizeof(float));
    cudaMalloc(&d_B, B_size * sizeof(float));
    cudaMalloc(&d_C, C_size * sizeof(float));

    cudaMemcpy(d_A, h_A, A_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_size * sizeof(float), cudaMemcpyHostToDevice);

    matrix_multiply<<<blocks, threads>>>(d_A, d_B, d_C, N, K, M);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, C_size * sizeof(float), cudaMemcpyDeviceToHost;

    print_array_vals(h_C, C_size);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}