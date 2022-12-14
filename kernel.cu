#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "err_handling.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <nvtx3/nvToolsExt.h>
#include <time.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define TILED_LEN 4096

int WatsonPairCheck(int t, int j, char* RNA);
__host__ __device__ void Nussinov(char* RNA, int N);
cudaError_t globalNussinovCuda(char* RNA, int N);
cudaError_t tiledNussinovCuda(char* RNA, int N);
void printCPURows(int** OPT, int N, int numOfRows);
void printGPURows(int* optGPU, int N, int numOfRows);

// now change to 1D block & 1D grid to do the parallization
__global__ void NussinovGlobal(int* d_optGPU, char* d_rna, int curOPTCol, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = curOPTCol;

    //printf("row:%d, col:%d, blockIdx.y:%d, blockDim.y:%d, threadIdx.y:%d \n", row, col, blockIdx.y, blockDim.y, threadIdx.y);

    if (row <= col && col < N)
    {
        if (row >= col - 4)
        {
            d_optGPU[row * N + col] = 0;
        }
        else
        {
            int exclude = d_optGPU[row * N + col - 1]; // OPT[i][j - 1];
            int include = 0;
            for (int t = row; t < col - 4; t++)
            {
                int result = WatsonPairCheck(t, col, d_rna);
                if (result == 1)
                {
                    int x = t - 1;
                    if (t - 1 < 0) x = N - 1;
                    int pair = 1 + d_optGPU[row * N + x] + d_optGPU[(t + 1) * N + col - 1];
                    //int pair = 1 + OPT[row][x] + OPT[t+1][col-1];              
                    include = MAX(include, pair);
                }
            }
            d_optGPU[row * N + col] = MAX(exclude, include);
        }

    }
}



__global__ void NussinovTiled(int* d_optGPU, char* d_rna, int curOPTCol, int N)
{                                        // OPT[i][j] needed element
    __shared__ int tiled_row[TILED_LEN]; // for OPT[i][t-1] 
    __shared__ int tiled_col[TILED_LEN]; // for OPT[t+1][j-1] 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = curOPTCol;

    //printf("row:%d, col:%d, blockIdx.y:%d, blockDim.y:%d, threadIdx.y:%d \n", row, col, blockIdx.y, blockDim.y, threadIdx.y);

    if (row <= col && col < N)
    {

        for (int sub = 0; sub < col; ++sub) tiled_row[0] = d_optGPU[row * N + sub];
        __syncthreads();
        for (int sub = row; sub < col && sub < TILED_LEN; ++sub) tiled_col[sub] = d_optGPU[sub * N + col - 1];
        __syncthreads();

        if (row >= col - 4)
        {
            d_optGPU[row * N + col] = 0;
        }
        else
        {

            int exclude = d_optGPU[row * N + col - 1]; // OPT[i][j - 1];
            int include = 0;
            // get the value from tiled
            for (int t = row; t < col - 4; t++)
            {
                int result = WatsonPairCheck(t, col, d_rna);
                if (result == 1)
                {
                    int x = t - 1;
                    if (t - 1 < 0) x = N - 1;
                    //int pair = 1 + tiled_row[x-row] + tiled_col[t+1];
                    int pair = 1 + d_optGPU[row * N + x];
                    if (t < TILED_LEN - 1) pair += tiled_col[t + 1];
                    else pair += d_optGPU[(t + 1) * N + col - 1];
                    //int pair = 1 + OPT[row][t-1] + OPT[t+1][col-1];


                    include = MAX(include, pair);
                }

            }
            __syncthreads();

            // get the result
            d_optGPU[row * N + col] = MAX(exclude, include);
        }
    }
}


int main()
{

    nvtxInitialize(0);
    int testcase = 10;

    int len[] = { 48, 128,512, 1024, 2048, 4096 ,8192,16384, 25600,32768 };
    char* s[] = { "rna_48.txt", "rna_128.txt","rna_512.txt","rna_1024.txt","rna_2048.txt", "rna_4096.txt", "rna_8192.txt","rna_16384.txt","rna_25600.txt","rna_32768.txt" };

    clock_t t_cpu_start, t_cpu_end, t_gpu_start, t_gpu_end, t_gpu_tiled_start, t_gpu_tiled_end;

    for (int i = 0; i < testcase; i++)
    {
        FILE* fptr;
        fptr = fopen(s[i], "r");
        if (fptr == NULL) {
            printf("Error! opening file\n");
            exit(1);
        }
        int N = len[i];
        char* rna = (char*)malloc((N + 1) * sizeof(char));
        fgets(rna, N + 1, fptr);
        // printf("The read rna:\n");
        // printf("%s\n", rna);
        fclose(fptr);


        printf("Current testing length = %d\n", N);
        // execute cpu only under 10000 since after that take more than 3hr to execute, which aws appstream using time cannot make it
        if (N < 10000)    
        {
            printf("Executing CPU function...\n");
            nvtxRangePush("Start: CPU_nussinov alogrithm");
            t_cpu_start = clock();
            Nussinov(rna, N);
            t_cpu_end = clock();
            nvtxRangePop();
            nvtxMark("CPU completed");
            double time_taken_cpu = (double)(t_cpu_end - t_cpu_start) / CLOCKS_PER_SEC;
            printf("time taken for CPU: %f\n", time_taken_cpu);
        }


        printf("Executing GPU global kernel...\n");
        nvtxRangePush("Start: GPU_nussinov alogrithm");
        t_gpu_start = clock();
        cudaError_t cudaStatus = globalNussinovCuda(rna, N);
        t_gpu_end = clock();
        nvtxRangePop();
        nvtxMark("GPU completed");
        double time_taken_gpu = (double)(t_gpu_end - t_gpu_start) / CLOCKS_PER_SEC;

        printf("time taken for GPU global kerenl: %f\n", time_taken_gpu);
        checkCuda(cudaStatus);

        printf("Executing GPU tiled kernel...\n");
        nvtxRangePush("Start: GPU_nussinov alogrithm");
        t_gpu_tiled_start = clock();
        cudaStatus = tiledNussinovCuda(rna, N);
        t_gpu_tiled_end = clock();
        nvtxRangePop();
        nvtxMark("GPU completed");
        double time_taken_gpu_tiled = (double)(t_gpu_tiled_end - t_gpu_tiled_start) / CLOCKS_PER_SEC;

        printf("time taken for GPU tiled kernel: %f\n", time_taken_gpu_tiled);
        checkCuda(cudaStatus);

        free(rna);
        printf("End current session\n");
        printf("\n");

    }

    return 0;
}


// reference: https://stackoverflow.com/questions/18315796/use-cpu-function-in-cuda
__host__ __device__ int WatsonPairCheck(int t, int j, char* RNA)
{
    if ((RNA[t] == 'A' && RNA[j] == 'U') || (RNA[t] == 'U' && RNA[j] == 'A') || (RNA[t] == 'C' && RNA[j] == 'G') || (RNA[t] == 'G' && RNA[j] == 'C'))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void Nussinov(char* RNA, int N)
{
    int** OPT = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++)
        OPT[i] = (int*)malloc(N * sizeof(int));


    // initialize the matrix
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            OPT[i][j] = 0;
        }
    }

    // ---------------- dp part ----------------
    for (int j = 0; j < N; j++)
    {
        for (int i = j; i > -1; i--)
        {
            if (i >= j - 4)
            {
                OPT[i][j] = 0;
            }
            else
            {
                int exclude = OPT[i][j - 1];
                int include = 0;
                for (int t = i; t < j - 4; t++)
                {
                    int result = WatsonPairCheck(t, j, RNA);

                    if (result == 1)
                    {
                        int x = t - 1;
                        if (t - 1 < 0) x = N - 1;
                        int pair = 1 + OPT[i][x] + OPT[t + 1][j - 1];

                        include = MAX(include, pair);
                    }
                }


                OPT[i][j] = MAX(exclude, include);
            }
        }
    }

    // Time the usage
    printf("maximum from CPU: %d\n", OPT[0][N - 1]);

    // print the table
    // printCPURows(OPT, N, 48);

    for (int i = 0; i < N; i++)
    {
        free(OPT[i]);
    }
    return;
}

void printCPURows(int** OPT, int N, int numOfRows)
{
    printf("the cpu opt table of %d rows\n", numOfRows);
    for (int i = 0; i < numOfRows; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", OPT[i][j]);
        }
        printf("\n");
    }
}

// references:
// https://medium.com/analytics-vidhya/matrix-multiplication-in-cuda-a-simple-guide-bab44bc1f8ab
// https://github.com/charitha22/workspace/blob/master/cuda/mm/naive_matrix_multiply.cu
cudaError_t globalNussinovCuda(char* RNA, int N)
{
    int* optGPU;
    cudaMallocHost((void**)&optGPU, sizeof(int) * N * N);

    int* d_optGPU = 0;
    char* d_rna = '\0';

    cudaError_t cudaStatus;
    cudaStatus = checkCuda(cudaSetDevice(0));
    checkCuda(cudaStatus);

    cudaStatus = cudaMalloc((int**)&d_optGPU, sizeof(int) * N * N);
    checkCuda(cudaStatus);
    cudaStatus = cudaMalloc((char**)&d_rna, sizeof(char) * N);
    checkCuda(cudaStatus);

    cudaStatus = cudaMemcpy(d_optGPU, optGPU, N * N * sizeof(int), cudaMemcpyHostToDevice);
    checkCuda(cudaStatus);
    cudaStatus = cudaMemcpy(d_rna, RNA, N * sizeof(char), cudaMemcpyHostToDevice);
    checkCuda(cudaStatus);


    for (int curOPTCol = 0; curOPTCol < N; curOPTCol++)
    {
        // only 1D since has to be calculate separately
        int numOfBlock = curOPTCol + 1;
        int blockSize, gridSize;
        if (numOfBlock <= 1024) blockSize = numOfBlock;
        else blockSize = 1024;


        gridSize = numOfBlock / 1024 + 1;
        if (numOfBlock % 1024 == 0)
            gridSize--;
        // printf("numOfBlock:%d, gridSize:%d, blockSize:%d", numOfBlock, gridSize, blockSize);

        dim3 dimBlock(1, blockSize, 1);
        dim3 dimGrid(1, gridSize, 1);

        NussinovGlobal << <dimGrid, dimBlock >> > (d_optGPU, d_rna, curOPTCol, N);

        cudaStatus = cudaGetLastError();
        checkCuda(cudaStatus);
        cudaStatus = cudaDeviceSynchronize();
        checkCuda(cudaStatus);
    }

    cudaStatus = cudaMemcpy(optGPU, d_optGPU, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    checkCuda(cudaStatus);
    cudaStatus = cudaMemcpy(d_rna, d_rna, N * sizeof(char), cudaMemcpyDeviceToHost);
    checkCuda(cudaStatus);


    // printf number of GPU rows
    //printGPURows(optGPU, N, 48);

    printf("maximum from GPU: %d\n", optGPU[N - 1]);

Error:
    cudaFree(d_optGPU);
    return cudaStatus;

}

cudaError_t tiledNussinovCuda(char* RNA, int N)
{

    int* optGPU;
    cudaMallocHost((void**)&optGPU, sizeof(int) * N * N);

    int* d_optGPU = 0;
    char* d_rna = '\0';

    cudaError_t cudaStatus;
    cudaStatus = checkCuda(cudaSetDevice(0));
    checkCuda(cudaStatus);

    cudaStatus = cudaMalloc((int**)&d_optGPU, sizeof(int) * N * N);
    checkCuda(cudaStatus);
    cudaStatus = cudaMalloc((char**)&d_rna, sizeof(char) * N);
    checkCuda(cudaStatus);

    cudaStatus = cudaMemcpy(d_optGPU, optGPU, N * N * sizeof(int), cudaMemcpyHostToDevice);
    checkCuda(cudaStatus);
    cudaStatus = cudaMemcpy(d_rna, RNA, N * sizeof(char), cudaMemcpyHostToDevice);
    checkCuda(cudaStatus);

    for (int curOPTCol = 0; curOPTCol < N; curOPTCol++)
    {
        // only 1D since has to be calculate separately
        int numOfBlock = curOPTCol + 1;
        int blockSize, gridSize;
        if (numOfBlock <= 1024) blockSize = numOfBlock;
        else blockSize = 1024;


        gridSize = numOfBlock / 1024 + 1;
        if (numOfBlock % 1024 == 0)
            gridSize--;
        // printf("numOfBlock:%d, gridSize:%d, blockSize:%d", numOfBlock, gridSize, blockSize);

        dim3 dimBlock(1, blockSize, 1);
        dim3 dimGrid(1, gridSize, 1);

        NussinovTiled << <dimGrid, dimBlock >> > (d_optGPU, d_rna, curOPTCol, N);
        cudaStatus = cudaGetLastError();
        checkCuda(cudaStatus);
        cudaStatus = cudaDeviceSynchronize();
        checkCuda(cudaStatus);
    }

    cudaStatus = cudaMemcpy(optGPU, d_optGPU, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    checkCuda(cudaStatus);
    cudaStatus = cudaMemcpy(d_rna, d_rna, N * sizeof(char), cudaMemcpyDeviceToHost);
    checkCuda(cudaStatus);


    // printf number of GPU rows
    //printGPURows(optGPU, N, 48);

    printf("maximum from GPU: %d\n", optGPU[N - 1]);

Error:
    cudaFree(d_optGPU);
    return cudaStatus;

}

void printGPURows(int* optGPU, int N, int numOfRows)
{
    printf("the gpu opt first %d rows \n", numOfRows);
    for (int i = 0; i < N * numOfRows; i++)
    {
        printf("%d ", optGPU[i]);
        if (i != 0 && (i + 1) % N == 0)
        {
            printf("\n");
        }
    }
}