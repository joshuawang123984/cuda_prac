#include <cuda_runtime.h>
#include <cstdlib>

__global__ void prefixSum(float *arr, int N, float *out)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        sdata[tid] = arr[i];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int idx = (tid + 1) * 2 * s - 1;
        if (idx < blockDim.x)
        {
            sdata[idx] += sdata[idx - s];
        }
        __syncthreads();
    }

    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        int idx = (tid + 1) * 2 * s - 1;
        if (idx + s < blockDim.x)
        {
            sdata[idx + s] += sdata[idx];
        }

        __syncthreads();
    }

    if (i < N)
        out[i] = sdata[tid];
}