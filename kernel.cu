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
// #include <unistd.h> 
// #include <sys/time.h>
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
//#define N 128
//#define N 1024
//#define N 12
#define BLOCK_SIZE 32
// float timedifference_msec(struct timeval t0, struct timeval t1);
int WatsonPairCheck(int t, int j, char* RNA);
__host__ __device__ void Nussinov(char* RNA, int N);
cudaError_t globalNussinovCuda(char* RNA, int N);
void printCPURows(int** OPT, int N, int numOfRows);
void printGPURows(int* optGPU, int N, int numOfRows);
// fill all the matrix with 0
__global__ void fillTheMatrix(int* d_optGPU, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        d_optGPU[row * N + col] = 0;
    }
}

// now change to 1D block & 1D grid to do the parallization
__global__ void NussinovGlobal(int* d_optGPU, char* d_rna, int curOPTCol, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = curOPTCol;
    //if (blockIdx.y > 0)
    //printf("row:%d, col:%d, blockIdx.y:%d, blockDim.y:%d, threadIdx.y:%d \n", row, col, blockIdx.y, blockDim.y, threadIdx.y);

    //if (row < N && col < N) // maybe can try and see the correctness, then no need to allocate different size block
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


int main()
{

    nvtxInitialize(0);
    nvtxMark("Hello world!");
    int testcase = 6;
    //int len[] = { 1024,2048,4096 };
    //char* s[] = { "rna_1024.txt", "rna_2048.txt", "rna_4096.txt" };
    int len[] = {1024, 2048, 4096 ,8192,16384, 25600 };
    char* s[] = {"rna_1024.txt","rna_2048.txt", "rna_4096.txt", "rna_8192.txt","rna_16384.txt","rna_25600.txt"};
    clock_t t_cpu_start, t_cpu_end, t_gpu_start, t_gpu_end;

    for (int i = 0; i < testcase; i++)
    {
        FILE* fptr;
        fptr = fopen(s[i], "r");
        if (fptr == NULL) {
            printf("Error! opening file\n");
            // Program exits if the file pointer returns NULL.
            exit(1);
        }
        int N = len[i];
        char* rna = (char*)malloc((N + 1) * sizeof(char));
        fgets(rna, N + 1, fptr);
        // printf("The read rna:\n");
        // printf("%s\n", rna);
        fclose(fptr);

        //for (int i = 0; i < N + 1; i++) printf("%c", rna[i]);
        printf("Current testing length = %d\n", N);
        printf("Executing CPU function...\n");
        nvtxRangePush("Start: CPU_nussinov alogrithm");
        t_cpu_start = clock();
        Nussinov(rna, N);
        t_cpu_end = clock();
        nvtxRangePop();
        nvtxMark("CPU completed");
        double time_taken_cpu = (double)(t_cpu_end - t_cpu_start) / CLOCKS_PER_SEC;
       
        printf("time taken for CPU: %f\n", time_taken_cpu);

        printf("Executing GPU kernel...\n");
        nvtxRangePush("Start: GPU_nussinov alogrithm");
        t_gpu_start = clock();
        cudaError_t cudaStatus = globalNussinovCuda(rna, N);
        t_gpu_end = clock();
        nvtxRangePop();
        nvtxMark("GPU completed");
        double time_taken_gpu = (double)(t_gpu_end - t_gpu_start) / CLOCKS_PER_SEC;
       
        printf("time taken for GPU: %f\n", time_taken_gpu);
        checkCuda(cudaStatus);
        free(rna);
        printf("End current session\n");
        printf("\n");

    }




    //printf("time spent\n");
    // 
    // printf("traceback pairs?");

    // TODO: the tiled version of Nussionv

    return 0;
}

/*
float timedifference_msec(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
}
*/

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
    // time reference:
    // https://stackoverflow.com/questions/10192903/time-in-milliseconds-in-c
    int** OPT = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++)
        OPT[i] = (int*)malloc(N * sizeof(int));

    // int* OPT[N];
    // for (int i = 0; i < N; i++)
    //{
    //    OPT[i] = (int*)malloc(N * sizeof(int));
    //}



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

    // TODO: call traceback function
    // traceback(0, N - 1, OPT, RNA);    

    // TODO: print out structure

    // print the table
    //printCPURows(OPT, N, 1);


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
// the github I gave Christine
cudaError_t globalNussinovCuda(char* RNA, int N)
{
    // TODO: remove optGPU parameter, initialize the OPT using kernel function
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
    // ASK: how to avoid data transfer from CPU to GPU again and again?
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
        //dim3 dimBlock(1, curOPTCol + 1, 1);
        // dim3 dimGrid(1, (curOPTCol - 1) / BLOCK_SIZE + 1, 1);
       // dim3 dimGrid(1, 1, 1);

        //printf("curOPTCol:%d", curOPTCol);
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
    //printGPURows(optGPU, N, 1);

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