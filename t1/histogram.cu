#include <cuda_runtime.h>
#include <cstdlib>

#define NUM_BINS 256

__global__ void histogram(int *values, int N, int *histogram)
{

    __shared__ int sdata[NUM_BINS];

    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    for (int b = tid; b < NUM_BINS; b += blockDim.x)
    {
        sdata[b] = 0;
    }

    __syncthreads();

    for (int idx = i; idx < N; idx += gridDim.x * blockDim.x)
    {
        int val = values[idx];
        atomicAdd(&sdata[val], 1);
    }
    __syncthreads();

    for (int b = tid; b < NUM_BINS; b += blockDim.x)
    {
        atomicAdd(&histogram[b], sdata[b]);
    }
}