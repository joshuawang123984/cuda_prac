#include <cuda_runtime.h>
#include <cstdlib>

#define TILE 16

__global__ void batchedgemm(float *A, float *B, float *C,
                            int m, int n, int k, int batch_size)
{
    int batch = blockIdx.z;

    int strideA = m * k;
    int strideB = k * n;
    int strideC = m * n;

    float *A_batch = A + batch * strideA;
    float *B_batch = B + batch * strideB;
    float *C_batch = C + batch * strideC;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
    {
        As[threadIdx.y][threadIdx.x] = A_batch[row * k + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B_batch[(t * TILE + threadIdx.y) * n + col];

        __syncthreads();

        for (int i = 0; i < TILE; ++i)
        {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    C_batch[row * n + col] = sum;
}