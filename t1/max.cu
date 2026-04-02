#include <cuda_runtime.h>
#include <cstdlib>

__global__ void max(float *arr, int N, float *max)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_max = -1e-20f;

    for (int idx = i; idx < N; idx += blockDim.x * gridDim.x)
    {
        local_max = fmaxf(local_max, arr[idx]);
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        max[blockIdx.x] = sdata[0];
    }
}