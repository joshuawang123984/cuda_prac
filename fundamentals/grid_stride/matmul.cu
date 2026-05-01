#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <random>

#include "functions.h"

#define TILE 4

__global__ void matmul(float *A, float *B, float *C, int N, int K, int M)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tile_row = blockIdx.y * TILE;
    int tile_col = blockIdx.x * TILE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = tile_row + ty;
    int col = tile_col + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K - 1) / TILE + 1; ++t)
    {
        int phase = t * TILE;

        if (row < N && (phase + tx) < K)
            As[ty][tx] = A[row * K + (phase + tx)];
        else
            As[ty][tx] = 0.0f;

        if (col < M && (phase + ty) < M)
            Bs[ty][tx] = B[(phase + ty) * M + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TIE; ++k)
        {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }
    if (row < N && col < M)
        C[row * M + col] = sum;
}

int main()
{
    int N = 64;
    int K = 16;
    int M = 32;
    int A_size = N * K;
    int B_size = K * M;
    int C_size = N * M;
    dim3 threads(TILE, TILE);
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

    matmul<<<blocks, threads>>>(d_A, d_B, d_C, N, K, M);

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