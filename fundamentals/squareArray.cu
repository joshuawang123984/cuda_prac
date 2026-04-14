#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <random>

__global__ void square_array(float *arr, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N)
        return;

    arr[idx] *= arr[idx];
}

void print_array_vals(float *arr, int N)
{
    for (int i = 0; i < N; ++i)
    {
        std::cout << arr[i] << " ";
    }

    std::cout << std::endl;
}

void initialize_random_vals(float *arr, int N)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 99);

    for (int i = 0; i < N; i++)
    {
        arr[i] = dist(gen);
    }
}

int main()
{
    int N = 32;
    int threads = 8;
    int blocks = (N - 1) / threads + 1;

    float *h_arr = (float *)malloc(N * sizeof(float));
    initialize_random_vals(h_arr, N);
    print_array_vals(h_arr, N);

    float *d_arr;
    cudaMalloc(&d_arr, N * sizeof(float));
    cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);

    square_array<<<blocks, threads>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost);

    print_array_vals(h_arr, N);

    cudaFree(d_arr);
    free(h_arr);
}