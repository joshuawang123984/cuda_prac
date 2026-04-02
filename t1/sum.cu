#include <cuda_runtime.h>
#include <cstdlib>

__global__ void sum(float *arr, int N, float *sum)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;

    for (int idx = i; idx < N; idx += gridDim.x * blockDim.x)
    {
        local_sum += arr[idx];
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        sum[blockIdx.x] = sdata[0];
    }
}