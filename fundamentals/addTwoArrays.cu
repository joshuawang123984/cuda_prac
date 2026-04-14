#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <random>

// asume lengths are same for both arrays
__global__ void add_two_arrays(float *arr1, float *arr2, float *sum_array, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // potential optimization, loop through a stride for i < N, each thread does multiple ops
    if (idx >= N)
    {
        return;
    }

    sum_array[idx] = arr1[idx] + arr2[idx];
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

void print_array_vals(float *arr, int N)
{
    for (int i = 0; i < N; ++i)
    {
        std::cout << arr[i] << " ";
    }

    std::cout << std::endl;
}

int main()
{
    int N = 32;
    int threads = 8;
    int blocks = (N - 1) / threads + 1;

    float *h_arr1 = (float *)malloc(N * sizeof(float));
    float *h_arr2 = (float *)malloc(N * sizeof(float));
    float *h_sum_array = (float *)malloc(N * sizeof(float));

    // can fuse these but program small so it doesnt matter
    initialize_random_vals(h_arr1, N);
    initialize_random_vals(h_arr2, N);

    print_array_vals(h_arr1, N);
    print_array_vals(h_arr2, N);

    float *d_arr1;
    float *d_arr2;
    float *d_sum_array;
    cudaMalloc(&d_arr1, N * sizeof(float));
    cudaMalloc(&d_arr2, N * sizeof(float));
    cudaMalloc(&d_sum_array, N * sizeof(float));

    cudaMemcpy(d_arr1, h_arr1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, h_arr2, N * sizeof(float), cudaMemcpyHostToDevice);

    add_two_arrays<<<blocks, threads>>>(d_arr1, d_arr2, d_sum_array, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_sum_array, d_sum_array, N * sizeof(float), cudaMemcpyDeviceToHost);

    print_array_vals(h_sum_array, N);

    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_sum_array);
    free(h_arr1);
    free(h_arr2);
    free(h_sum_array);
    return 0;
}