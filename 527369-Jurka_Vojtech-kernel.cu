#define MAX_BLOCK 64


__global__ void compute(sGalaxy A, sGalaxy B, int n, float* res)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int tile_size = blockDim.x;

    float axi = A.x[i];
    float ayi = A.y[i];
    float azi = A.z[i];
    float bxi = B.x[i];
    float byi = B.y[i];
    float bzi = B.z[i];
    
    __shared__ float Ax[MAX_BLOCK];
    __shared__ float Ay[MAX_BLOCK];
    __shared__ float Az[MAX_BLOCK];
    __shared__ float Bx[MAX_BLOCK];
    __shared__ float By[MAX_BLOCK];
    __shared__ float Bz[MAX_BLOCK];
    
    float tmp = 0.0f;

    for (int b = bx; b < n/tile_size + 1; b++)
    {
        if (i < n)
        {
            Ax[tx] = A.x[b*tile_size + tx];
            Ay[tx] = A.y[b*tile_size + tx];
            Az[tx] = A.z[b*tile_size + tx];
            Bx[tx] = B.x[b*tile_size + tx];
            By[tx] = B.y[b*tile_size + tx];
            Bz[tx] = B.z[b*tile_size + tx];
        }
        
        
        __syncthreads();
        
        for (int j = 0; j < tile_size; j++)
        {
            if (b*tile_size+j > i && b*tile_size+j < n)
            {
                float da = sqrt((axi-Ax[j])*(axi-Ax[j])
                    + (ayi-Ay[j])*(ayi-Ay[j])
                    + (azi-Az[j])*(azi-Az[j]));
                float db = sqrt((bxi-Bx[j])*(bxi-Bx[j])
                    + (byi-By[j])*(byi-By[j])
                    + (bzi-Bz[j])*(bzi-Bz[j]));
                tmp += (da-db) * (da-db);
            }
        }
        __syncthreads();
    }

    if(i < n-1)
    {
        res[i] = tmp;
    }

}

float solveGPU(sGalaxy A, sGalaxy B, int n) 
{
    size_t size = n * sizeof(float);
    float* h_res = (float*)malloc(size);
    float* d_res;
    cudaMalloc(&d_res, size);

    int block = 32;
    
    compute<<<n/block+1, block>>>(A, B, n, d_res);

    cudaMemcpy(h_res, d_res, size, cudaMemcpyDeviceToHost);
    

    float res = 0.0f;
    for (int i = 0; i < n-1; i++) 
    {
        res += h_res[i];
    }

    res = sqrt(1/((float)n*((float)n-1)) * res);
    
    cudaFree(d_res);
    free(h_res);

    return res;
}