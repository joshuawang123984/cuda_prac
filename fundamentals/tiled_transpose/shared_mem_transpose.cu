#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <random>

#include "functions.h"

#define TILE 16

__global__ void transpose(float *A, float *B, int R, int C)
{
    __shared__ float sdata[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    if (row < R && col < C)
    {
        tile[threadIdx.y][threadIdx.x] = A[row * C + col];
    }

    __syncthreads();

    int newRow = blockIdx.x * TILE + threadIdx.y;
    int newCol = blockIdx.y * TILE + threadIdx.x;

    if (newRow < C && newCol < R)
    {
        B[newRow * R + newCol] = tile[threadIdx.x][threadIdx.y];
    }
}

int main()
{
    int R = 32;
    int C = 32;
    int N = R * C;
    dim3 threads(TILE, TILE);
    dim3 blocks((C + TILE - 1) / TILE, (R + TILE - 1) / TILE);

    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));

    initialize_random_vals(h_A, N);
    print_array_vals(h_A, N);

    float *d_A;
    float *d_B;

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);

    transpose<<<blocks, threads>>>(d_A, d_B, R, C);

    cudaMemcpy(h_B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost);
    print_array_vals(h_B, N);

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}