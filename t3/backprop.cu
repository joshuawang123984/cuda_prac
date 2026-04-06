#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <math.h>

#include "gemm.h"
#include "softmax.h"

#define BLOCK_SIZE 256

__global__ void subtract(float *dZ, float *Y_hat, float *Y, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        dZ[i] = Y_hat[i] - Y[i];
}

__global__ void reduce(float *dZ, float *db, int R, int C)
{
    int col = blockIdx.x;
    int tid = threadIdx.x;

    if (col >= C)
        return;

    __shared__ float sdata[BLOCK_SIZE];

    float sum = 0.0f;

    for (int i = tid; i < R; i += blockDim.x)
        sum += dZ[i * C + col];

    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        db[col] = sdata[0];
}

int main()
{
    int R = 128, K = 64, C = 10;

    size_t sizeX = R * K * sizeof(float);
    size_t sizeW = K * C * sizeof(float);
    size_t sizeZ = R * C * sizeof(float);

    // -------------------------
    // Host memory
    // -------------------------
    float *h_X = (float *)malloc(sizeX);
    float *h_W = (float *)malloc(sizeW);
    float *h_Y = (float *)malloc(sizeZ);
    float *h_dW = (float *)malloc(sizeW);

    // Initialize (important)
    for (int i = 0; i < R * K; i++)
        h_X[i] = 0.01f;
    for (int i = 0; i < K * C; i++)
        h_W[i] = 0.01f;
    for (int i = 0; i < R * C; i++)
        h_Y[i] = 0.0f;

    // -------------------------
    // Device memory
    // -------------------------
    float *X, *W, *Z, *Y_hat, *Y;
    float *dZ, *dW, *db, *dX;

    cudaMalloc(&X, sizeX);
    cudaMalloc(&W, sizeW);
    cudaMalloc(&Z, sizeZ);
    cudaMalloc(&Y_hat, sizeZ);
    cudaMalloc(&Y, sizeZ);

    cudaMalloc(&dZ, sizeZ);
    cudaMalloc(&dW, sizeW);
    cudaMalloc(&db, C * sizeof(float));
    cudaMalloc(&dX, sizeX);

    // -------------------------
    // Host → Device
    // -------------------------
    cudaMemcpy(X, h_X, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(W, h_W, sizeW, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, h_Y, sizeZ, cudaMemcpyHostToDevice);

    // -------------------------
    // Forward
    // -------------------------
    dim3 block2D(16, 16);
    dim3 grid2D((C + 15) / 16, (R + 15) / 16);

    GEMM<<<grid2D, block2D>>>(X, W, Z, R, C, K);

    dim3 gridSoftmax(R);
    dim3 blockSoftmax(256);

    softmax<<<gridSoftmax, blockSoftmax>>>(Z, C, R);

    Y_hat = Z;

    // -------------------------
    // Backward
    // -------------------------
    int total = R * C;
    dim3 block1D(256);
    dim3 grid1D((total + 255) / 256);

    subtract<<<grid1D, block1D>>>(dZ, Y_hat, Y, total);

    transpose<<<grid2D, block2D>>>(X, X_T, R, K);
    GEMM<<<grid2D, block2D>>>(X_T, dZ, K, R, C, dW);

    dim3 gridReduce(C);
    reduce<<<gridReduce, block1D>>>(dZ, db, R, C);

    transpose<<<grid2D, block2D>>>(W, W_T, K, C);
    GEMM<<<grid2D, block2D>>>(dZ, W_T, R, C, K, dX);

    cudaDeviceSynchronize();

    // -------------------------
    // Device → Host
    // -------------------------
    cudaMemcpy(h_dW, dW, sizeW, cudaMemcpyDeviceToHost);

    std::cout << "First few dW values:\n";
    for (int i = 0; i < 10; i++)
        std::cout << h_dW[i] << " ";
    std::cout << std::endl;

    // -------------------------
    // Cleanup
    // -------------------------
    cudaFree(X);
    cudaFree(W);
    cudaFree(Z);
    cudaFree(Y_hat);
    cudaFree(Y);
    cudaFree(dZ);
    cudaFree(dW);
    cudaFree(db);
    cudaFree(dX);

    free(h_X);
    free(h_W);
    free(h_Y);
    free(h_dW);

    return 0;
}