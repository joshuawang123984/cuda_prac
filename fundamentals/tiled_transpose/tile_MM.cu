#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <random>

#include "functions.h"

#define TILE 8

__global__ void tile_matrix_multiply(float *A, float *B, float *C, int N, int K, int M)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum = 0.0f;

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
    {
        int tiledCol = t * TILE + threadIdx.x;
        int tiledRow = t * TILE + threadIdx.y;

        if (row < N && tiledCol < K)
        {
            As[threadIdx.y][threadIdx.x] = A[row * K + tiledCol];
        }
        else
        {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < M && tiledRow < K)
        {
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * M + col];
        }
        else
        {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE; ++i)
        {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < M)
        C[row * M + col] = sum;
}

int main()
{
    int N = 32;
    int K = 16;
    int M = 32;

    dim3 threads(TILE, TILE);
    dim3 blocks((M - 1) / threads.x + 1, (N - 1) / threads.y + 1);

    int A_size = N * K;
    int B_size = K * M;
    int C_size = N * M;

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

    tile_matrix_multiply<<<blocks, threads>>>(d_A, d_B, d_C, N, K, M);
    cudaDeviceSynchronize();
    cudaGetLastError();

    cudaMemcpy(h_C, d_C, C_size * sizeof(float), cudaMemcpyDeviceToHost);
    print_array_vals(h_C, C_size);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}