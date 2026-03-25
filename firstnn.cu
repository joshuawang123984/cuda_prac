enum ActivationType = {RELU, NONE};

#define TILE 16
#define WPT 2 // work per thread

__global__ void matmul_register_block(float *A, float *B, float *C, int N)
{

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE * WPT];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE * WPT + tx;

    float sum[WPT] = {0.0f, 0.0f}; // registers

    for (int t = 0; t < N; t += TILE)
    {

        // Load A tile
        As[ty][tx] = A[row * N + (t + tx)];

        // Load multiple B values (coalesced)
        for (int w = 0; w < WPT; w++)
        {
            Bs[ty][tx + w * TILE] =
                B[(t + ty) * N + col + w * TILE];
        }

        __syncthreads();

        // Compute
        for (int k = 0; k < TILE; k++)
        {
            float a = As[ty][k];

            for (int w = 0; w < WPT; w++)
            {
                sum[w] += a * Bs[k][tx + w * TILE];
            }
        }

        __syncthreads();
    }

    // Store results
    for (int w = 0; w < WPT; w++)
    {
        C[row * N + col + w * TILE] = sum[w];
    }
}

__global__ void bias_relu(float *x, float *b, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        float val = x[i] + b[i];
        x[i] = (val > 0) ? val : 0;
    }
}

__global__ void bias_add(float *x, float *b, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        x[i] += b[i];
    }
}

__global__ void softmax(float *x, int N)
{
    float sum = 0.0f;

    // compute exp + sum (single thread version for now)
    for (int i = 0; i < N; i++)
    {
        x[i] = expf(x[i]);
        sum += x[i];
    }

    for (int i = 0; i < N; i++)
    {
        x[i] /= sum;
    }
}
class FeedForward
{
public:
    int input_size, hidden_size, output_size;
    ActivationType activation;

    float *d_w1, *d_b1;
    float *d_w2, *d_b2;
    float *d_z1;

    FeedForward(int in, int hid, int out, ActivationType act)
        : input_size(in), hidden_size(hid), output_size(out), activation(act)
    {
        cudaMalloc(&d_w1, in * hid * sizeof(float));
        cudaMalloc(&d_b1, hid * sizeof(float));
        cudaMalloc(&d_w2, hid * out * sizeof(float));
        cudaMalloc(&d_b2, out * sizeof(float));
        cudaMalloc(&d_z1, hid * sizeof(float));

        float *h_w1 = initializeWeights(in, hid, d_w1);
        delete[] h_w1;

        float *h_w2 = initializeWeights(hid, out, d_w2);
        delete[] h_w2;

        cudaMemset(d_b1, 0, hid * sizeof(float));
        cudaMemset(d_b2, 0, out * sizeof(float));
    }

    ~FeedForward()
    {
        cudaFree(d_w1);
        cudaFree(d_b1);
        cudaFree(d_w2);
        cudaFree(d_b2);
        cudaFree(d_z1);
    }

    void forward(float *d_input, float *d_output)
    {
        matmul<<<grid1, block1>>>(
            d_input, d_w1, d_z1,
            1, input_size, hidden_size);

        int threads = 256;
        int blocks = (hidden_size + threads - 1) / threads;

        if (activation == RELU)
        {
            bias_relu<<<blocks, threads>>>(d_z1, d_b1, hidden_size);
        }

        matmul<<<grid2, block2>>>(
            d_z1, d_w2, d_output,
            1, hidden_size, output_size);

        blocks = (output_size + threads - 1) / threads;
        bias_add<<<blocks, threads>>>(d_output, d_b2, output_size);

        softmax<<<1, 1>>>(d_output, output_size);
    }

private:
    float *initializeWeights(int in, int out, float *param)
    {
        float *h_ = new float[in * out];
        for (int i = 0; i < in * out; i++)
        {
            h_[i] = ((float)rand() / RAND_MAX) * 0.01f;
        }

        cudaMemcpy(param, h_, in * out * sizeof(float), cudaMemcpyHostToDevice);

        return h_;
    }
};