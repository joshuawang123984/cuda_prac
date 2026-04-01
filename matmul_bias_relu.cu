#include <cuda_runtime.h>
#include <cstdlib>

#define TILE 16

__global__ void matmul_bias_relu(float *A, float *B, float *bias, float *C, int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N; t += TILE)
    {
        As[threadIdx.y][threadIdx.x] = A[row * N + (t + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * K + col];

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x]
        }

        __syncthreads();
    }

    if (row < M && col < K)
    {
        float val = sum + bias[col];
        C[row * K + col] = (val > 0) ? val : 0;
    }
}

__global__ void matmul_bias(float *A, float *B, float *bias, float *C,
                            int M, int N, int K)
{

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int k = 0; k < N; ++k)
    {
        sum += A[row * N + k] * B[k * K + col];
    }

    if (row < M && col < K)
    {
        C[row * K + col] = sum + bias[col];
    }
}

__global__ void softmax_kernel(float *x, int B, int N)
{
    int row = blockIdx.x;

    if (row >= B)
        return;

    float max_val = -1e20f;

    for (int i = 0; i < N; ++i)
    {
        float val = x[row * N + i];
        max_val = (max_val > val) ? max_val : val;
    }

    float sum = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        float e = expf(x[row * N + i] - max_val);
        x[row * N + i] = e;
        sum += e;
    }

    for (int i = 0; i < N; ++i)
    {
        x[row * N + i] /= sum;
    }
}