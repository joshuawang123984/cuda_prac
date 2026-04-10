#include <cuda_runtime.h>
#include <cstdlib>

#define TILE 16

__global__ void GEMM(float *A, float *B, int N, int K, int M, float *C)
{
    int tid = threadIdx.x;
    int i = BlockDim.x * BlockIdx.x + threadIdx.x;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
    {
        if (row < N && t * TILE + threadIdx.x < K)
        {
            As[threadIdx.y][threadIdx.x] = A[row * K + (t * TILE + threadIdx.x)];
        }
        else
        {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < M && t * TILE + threadIdx.y < K)
        {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) + M * col]
        }
        else
        {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < M)
    {
        C[row * M + col] = sum;
    }
}