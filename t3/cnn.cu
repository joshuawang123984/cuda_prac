#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <math.h>

#define BLOCK_SIZE 256
#define TILE 16
#define MAX_K 7

#define sqrt_2_over_pi = 0.79788456f
#define COEFF = 0.044715f
#define THREE_COEFF = 0.134145f

__global__ void GeLU(float *inputs, int R, int C)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= R)
        return;

    for (int i = tid; i < C; i += blockDim.x)
    {
        float x = inputs[row * C + i];
        float x3 = x * x * x;
        inputs[row * C + i] = 0.5f * x * (1.0f + tanh(sqrt_2_over_pi * (x + COEFF * x3)));
    }
}

__global__ void gelu_backward(float *dX, float *dY, float *input, int H, int W)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = H * W;

    if (idx >= total)
        return;

    float x = input[idx];
    float x2 = x * x;
    float x3 = x * x * x;

    float u = sqrt_2_over_pi * (x + COEFF * x3);
    float t = tanhf(u);

    float sech2 = 1.0f - t * t;

    float du_dx = SQRT_2_OVER_PI * (1.0f + THREE_COEFF * x2);

    float grad = 0.5f * (1.0f + t) +
                 0.5f * x * sech2 * du_dx;

    dX[idx] = dY[idx] * grad;
}

__global__ void softmax(float *x, int R, int C)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= R)
        return;

    __shared__ float sdata[BLOCK_SIZE];

    // find max
    // subtract by max then exponentiate
    // find sum
    // divide by sum

    float max_val = -1e20f;

    for (int i = tid; i < C; i += blockDim.x)
    {
        max_val = fmaxf(max_val, x[row * C + i]);
    }

    sdata[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    max_val = sdata[0];

    float sum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x)
    {
        float e = expf(x[row * C + i] - max_val);
        x[row * C + i] = e;
        sum += e;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    sum = sdata[0];
    for (int i = tid; i < C; i += blockDim.x)
    {
        x[row * C + i] /= sum;
    }
}

__global__ void pool(float *input, float *output,
                     int R, int C,
                     int pool_size, int stride)
{
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    int R_out = R / stride;
    int C_out = C / stride;

    if (out_row >= R_out || out_col >= C_out)
        return;

    int in_row = out_row * stride;
    int in_col = out_col * stride;

    float max_val = -1e20f;

    for (int i = 0; i < pool_size; ++i)
    {
        for (int j = 0; j < pool_size; ++j)
        {
            float val = input[(in_row + i) * C + (in_col + j)];
            max_val = fmaxf(max_val, val);
        }
    }

    output[out_row * C_out + out_col] = max_val;
}

__global__ void maxpool2x2(float *input, float *output, int R, int C)
{
    int out_row = blockDim.y * blockIdx.y + threadIdx.y;
    int out_col = blockDim.x * blockIdx.x + threadIdx.x;

    int R_out = R / 2;
    int C_out = C / 2;

    if (out_row >= R_out || out_col >= C_out)
        return;

    int in_row = out_row * 2;
    int in_col = out_col * 2;

    int idx = in_row * C + in_col;

    float a = input[idx];
    float b = input[idx + 1];
    float c = input[idx + C];
    float d = input[idx + C + 1];

    float max_val = fmaxf(fmaxf(a, b), fmaxf(c, d));
    output[out_row * C_out + out_col] = max_val;
}

__global__ void maxpool4d(float *input, float *output,
                          int N, int C, int H, int W)
{
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc % C;

    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    int H_out = H / 2;
    int W_out = W / 2;

    if (out_row >= H_out || out_col >= W_out)
        return;

    int in_row = out_row * 2;
    int in_col = out_col * 2;

    int base = ((n * C + c) * H + in_row) * W + in_col;

    float a = input[base];
    float b = input[base + 1];
    float c1 = input[base + W];
    float d = input[base + W + 1];

    float max_val = fmaxf(fmaxf(a, b), fmaxf(c1, d));

    int out_idx = ((n * C + c) * H_out + out_row) * W_out + out_col;
    output[out_idx] = max_val;
}

__global__ void subtract(float *dZ, const float *Y_hat, float *Y, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
    {
        dZ[i] = Y_hat[i] - Y[i];
    }
}

__global__ void conv2d(float *input,  // [N, Cin, H, W]
                       float *weight, // [Cout, Cin, K, K]
                       float *output, // [N, Cout, H_out, W_out]
                       int N, int Cin, int H, int W,
                       int Cout, int K)
{

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int h = blockIdx.y * TILE + ty;
    int w = blockIdx.x * TILE + tx;

    int z = blockIdx.z;
    int n = z / Cout;
    int cout = z % Cout;

    __shared__ float tile[TILE + MAX_K - 1][TILE + MAX_K - 1];

    float sum = 0.0f;
    for (int cin = 0; cin < Cin; ++cin)
    {
        for (int i = ty; i < TILE + K - 1; i += blockDim.y)
        {
            for (int j = tx; j < TILE + K - 1; j += blockDim.x)
            {
                int in_h = blockIdx.y * TILE + i;
                int in_w = blockIdx.x * TILE + j;

                if (in_h < H && in_w < W)
                {
                    tile[i][j] = input[((n * Cin + cin) * H + in_h) * W + in_w];
                }
            }
        }

        __syncthreads();

        if (h < H_out && w < W_out)
        {
            for (int i = 0; i < K; ++i)
            {
                for (int j = 0; j < K; ++j)
                {
                    float x = tile[ty + i][tx + j];
                    float wgt = weight[((cout * Cin + cin) * K + i) * K + j];
                    sum += x * wgt;
                }
            }
        }

        __syncthreads();
    }

    if (h < H_out && w < W_out)
    {
        int out_idx = ((n * Cout + cout) * H_out + h) * W_out + w;
        output[out_idx] = sum;
    }
}

__global__ void im2col(float *input, float *output, int Cin, int H, int W, int K)
{
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    if (h >= H_out || w >= W_out)
        return;

    int patch_idx = h * W_out + w;

    for (int cin = 0; cin < Cin; ++cin)
    {
        for (int i = 0; i < K; ++i)
        {
            for (int j = 0; j < K; ++j)
            {
                int col_idx = cin * K * K + i * K + j;
                float val = input[(cin * H + (h + i)) * W + (w + j)];
                output[patch_idx * (Cin * K * K) + col_idx] = val;
            }
        }
    }
}

__global__ void im2col_optimized(float *input, float *output, int Cin, int H, int W, int K)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H_out * W_out;

    if (idx >= total)
        return;

    int h = idx / W_out;
    int w = idx % W_out;

    int col_size = Cin * K * K;

    for (int cin = 0; cin < Cin; ++cin)
    {
        int base = (cin * H + h) * W + w;

        for (int i = 0; i < K; ++i)
        {
            int row_offset = base + i * W;

#pragma unroll
            for (int j = 0; j < K; ++j)
            {
                int col_idx = cin * K * K + i * K + j;
                output[idx * col_size + col_idx] = input[row_offset + j];
            }
        }
    }
}

__global__ void conv_backward(float *X, float *dZ, float *dW, int N, int Cin, int H, int W, int Cout, int K)
{
    int cout = blockIdx.x;
    int cin = blockIdx.y;
    int i = threadIdx.x;
    int j = threadIdx.y;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n)
    {
        for (int h = 0; h < H - K + 1; ++h)
        {
            for (int w = 0; w < W - K + 1; ++w)
            {
                float x = X[((n * Cin + cin) * H + (h + i)) * W + (w + j)];
                float dz = dZ[((n * Cout + cout) * (H - K + 1) + h) * (W - K + 1) + w];

                sum += x * dz;
            }
        }
    }

    int idx = ((cout * Cin + cin) * K + i) * K + j;
    dW[idx] = sum;
}

__global__ void pool_backward(float *dX, float *dY, int *mask, int N, int C, int H, int W, int H_out, int W_out, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = N * C * H_out * W_out;

    if (idx >= total)
        return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    float grad = dY[idx];

    int max_idx = mask[idx];

    dX[max_idx] = grad;
}

int main()
{
    //--------------------------------------------
    // Dimensions
    //--------------------------------------------
    int N = 1;          // batch size
    int Cin = 3;        // input channels
    int H = 32, W = 32; // input size

    int Cout = 8; // conv filters
    int K = 3;    // kernel size

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    //--------------------------------------------
    // Allocate device memory
    //--------------------------------------------
    float *d_input, *d_weight, *d_output;
    float *d_pool, *d_fc, *d_probs;
    float *dY, *dZ, *dW, *dX;
    int *d_mask;

    cudaMalloc(&d_input, N * Cin * H * W * sizeof(float));
    cudaMalloc(&d_weight, Cout * Cin * K * K * sizeof(float));
    cudaMalloc(&d_output, N * Cout * H_out * W_out * sizeof(float));

    int H_pool = H_out / 2;
    int W_pool = W_out / 2;

    cudaMalloc(&d_pool, N * Cout * H_pool * W_pool * sizeof(float));
    cudaMalloc(&d_mask, N * Cout * H_pool * W_pool * sizeof(int));

    int FC_in = Cout * H_pool * W_pool;
    int FC_out = 10;

    cudaMalloc(&d_fc, FC_in * FC_out * sizeof(float));
    cudaMalloc(&d_probs, N * FC_out * sizeof(float));

    cudaMalloc(&dY, N * FC_out * sizeof(float));
    cudaMalloc(&dZ, N * FC_out * sizeof(float));
    cudaMalloc(&dW, Cout * Cin * K * K * sizeof(float));
    cudaMalloc(&dX, N * Cin * H * W * sizeof(float));

    //--------------------------------------------
    // Kernel configs
    //--------------------------------------------
    dim3 block2D(16, 16);

    dim3 convGrid(
        (W_out + 15) / 16,
        (H_out + 15) / 16,
        N * Cout);

    dim3 poolGrid(
        (W_pool + 15) / 16,
        (H_pool + 15) / 16,
        N * Cout);

    dim3 softmaxGrid(N);
    dim3 softmaxBlock(256);

    dim3 block1D(256);
    dim3 grid1D((N * FC_out + 255) / 256);

    //--------------------------------------------
    // FORWARD PASS
    //--------------------------------------------

    // Conv
    conv2d<<<convGrid, block2D>>>(
        d_input, d_weight, d_output,
        N, Cin, H, W, Cout, K);

    // Activation
    GeLU<<<N * Cout, 256>>>(d_output, H_out, W_out);

    // Pool
    maxpool4d<<<poolGrid, block2D>>>(
        d_output, d_pool,
        N, Cout, H_out, W_out);

    // Flatten + FC (GEMM)
    // (treat d_pool as [N x FC_in])
    dim3 fcBlock(16, 16);
    dim3 fcGrid((FC_out + 15) / 16, (N + 15) / 16);

    GEMM<<<fcGrid, fcBlock>>>(
        d_pool, // [N x FC_in]
        d_fc,   // [FC_in x FC_out]
        N, FC_in, FC_out,
        d_probs); // [N x FC_out]

    // Softmax
    softmax<<<softmaxGrid, softmaxBlock>>>(
        d_probs, N, FC_out);

    cudaDeviceSynchronize();

    //--------------------------------------------
    // BACKWARD PASS
    //--------------------------------------------

    // dZ = Y_hat - Y
    subtract<<<grid1D, block1D>>>(
        dZ, d_probs, dY, N * FC_out);

    // dW_fc = X^T * dZ
    GEMM<<<fcGrid, fcBlock>>>(
        d_pool, dZ,
        FC_in, N, FC_out,
        d_fc);

    // dX_fc = dZ * W^T
    GEMM<<<fcGrid, fcBlock>>>(
        dZ, d_fc,
        N, FC_out, FC_in,
        d_pool);

    // Pool backward
    pool_backward<<<grid1D, block1D>>>(
        d_output, d_pool, d_mask,
        N, Cout, H_out, W_out,
        H_pool, W_pool, 2);

    // Activation backward
    gelu_backward<<<grid1D, block1D>>>(
        d_output, d_output, d_output,
        H_out, W_out);

    // Conv backward (dW)
    dim3 convBackGrid(Cout, Cin);
    dim3 convBackBlock(K, K);

    conv_backward<<<convBackGrid, convBackBlock>>>(
        d_input, d_output, dW,
        N, Cin, H, W, Cout, K);

    cudaDeviceSynchronize();

    //--------------------------------------------
    // Cleanup
    //--------------------------------------------
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaFree(d_pool);
    cudaFree(d_fc);
    cudaFree(d_probs);
    cudaFree(dY);
    cudaFree(dZ);
    cudaFree(dW);
    cudaFree(dX);
    cudaFree(d_mask);

    return 0;
}