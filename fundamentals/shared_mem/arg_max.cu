#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <random>

struct Pair
{
    float value;
    int index;
};

void initialize_random_pairs(Pair *A, int N)
{
    for (int i = 0; i < N; i++)
    {
        A[i].value = rand() % 100;
        A[i].index = i;
    }
}

void print_pair_vals(Pair *A, int N)
{
    for (int i = 0; i < N; ++i)
    {
        std::cout << "Index: " << A[i].index << " | Value: " << A[i].value << std::endl;
    }
}

__global__ void argmax(Pair *A, Pair *B, Pair *C, int N)
{
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ Pair sdata[256];

    if (i < N)
    {
        sdata[tid] = A[i].value > B[i].value ? A[i] : B[i];
    }
    else
    {
        sdata[tid] = {-1e20f, -1};
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] = sdata[tid].value > sdata[tid + stride].value ? sdata[tid] : sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        C[blockIdx.x] = sdata[0];
    }
}

int main()
{
    int N = 1024;
    int threads = 256;
    int blocks = (N - 1) / threads + 1;

    Pair *h_A = (Pair *)malloc(N * sizeof(Pair));
    Pair *h_B = (Pair *)malloc(N * sizeof(Pair));
    Pair *h_C = (Pair *)malloc(blocks * sizeof(Pair));

    initialize_random_pairs(h_A, N);
    initialize_random_pairs(h_B, N);

    print_pair_vals(h_A, N);
    print_pair_vals(h_B, N);

    Pair *d_A;
    Pair *d_B;
    Pair *d_C;

    cudaMalloc(&d_A, N * sizeof(Pair));
    cudaMalloc(&d_B, N * sizeof(Pair));
    cudaMalloc(&d_C, blocks * sizeof(Pair));

    cudaMemcpy(d_A, h_A, N * sizeof(Pair), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(Pair), cudaMemcpyHostToDevice);
    argmax<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, blocks * sizeof(Pair), cudaMemcpyDeviceToHost);
    print_pair_vals(h_C, blocks);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}