#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

__global__ void hello_threads(int *global_idxs, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N)
        return;

    global_idxs[idx] = idx;
}

int main()
{
    int N = 32;
    int threads = 8;
    int blocks = (N + threads - 1) / threads;

    int *h_arr = new int[N]();

    int *d_arr;
    cudaMalloc(&d_arr, N * sizeof(int));

    hello_threads<<<blocks, threads>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
    {
        std::cout << h_arr[i] << " ";
    }

    std::cout << std::endl;
    cudaFree(d_arr);
    delete[] h_arr;

    return 0;
}