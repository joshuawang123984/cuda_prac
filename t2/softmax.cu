#include <cuda_runtime.h>
#include <cstdlib>
#include <math.h>

__global__ void softmax(float *values, int N, int R)
{

    int row = blockIdx.x;

    if (row >= R)
        return;

    float max_val = -1e20f;

    for (int i = 0; i < N; ++i)
    {
        float val = values[row * N + i];
        max_val = (max_val > val) ? max_val : val;
    }

    float sum = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        float e = expf(values[row * N + i] - max_val);
        values[row * N + i] = e;
        sum += e;
    }

    for (int i = 0; i < N; ++i)
    {
        values[row * N + i] /= sum;
    }
}

#define BLOCK_SIZE 256

__global__ void optimizedsoftmax(float *x, int R, int C)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= R)
        return;

    __shared__ float sdata[BLOCK_SIZE];

    float max_val = -1e20f;

    for (int i = tid; i < C; i += blockDim.x)
    {
        float val = x[row * C + i];
        max_val = fmaxf(max_val, val);
    }

    sdata[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    max_val = sdata[0];

    float sum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x)
    {
        float e = expf(x[row * C + i] - max_val);
        x[row * C + i] = e;
        sum += e;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    sum = sdata[0];

    for (int i = tid; i < C; i += blockDim.x)
    {
        x[row * C + i] /= sum;
    }
}