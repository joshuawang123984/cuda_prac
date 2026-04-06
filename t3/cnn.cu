#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void GeLU(float *inputs, int R, int C)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= R)
        return;

    const float sqrt_2_over_pi = 0.79788456f;

    for (int i = tid; i < C; i += blockDim.x)
    {
        float x = inputs[row * C + i];
        float x3 = x * x * x;
        inputs[row * C + i] = 0.5f * x * (1.0f + tanh(sqrt_2_over_pi * (x + 0.044715f * x3)));
    }
}

__global__ void softmax(float *x, int R, int C)
{
    int row = blockIdx.x int tid = threadIdx.x;

    if (row >= R)
        return;

    __shared__ float sdata[BLOCK_SIZE];
    // find max
    // subtract by max then exponentiate
    // find sum
    // divide by sum
}

int main()
{

    //--------------------------------------------
    // FORWARD PASS
    //--------------------------------------------
    // input -> conv -> activation -> pool -> repeat? -> gemm/mm -> softmax
    conv<<<...>>>();
    GeLU<<<...>>>();
    pool<<<...>>>();

    conv<<<...>>>();
    GeLU<<<...>>>();
    pool<<<...>>>();

    GEMM<<<...>>>();
    softmax<<<...>>>();

    //--------------------------------------------
    // BACKWARD PASS
    //--------------------------------------------
    // dLoss -> softmax backward -> gemm/mm backward -> activation backward -> pool backward -> conv backward

    subtract<<<...>>>();
    GEMM<<<...>>>();
    GEMM<<<...>>>();

    pool_backward<<<...>>>();
    gelu_backward<<<...>>>();

    conv_backward<<<...>>>();
}