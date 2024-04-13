#include <cuda_runtime.h>
#include <stdio.h>
#include <cub/cub.cuh> 

const int M = 1024; //矩阵行
const int N = 2048; //矩阵列
const dim3 block_size(32, 32);
const dim3 grid_size(N/32, M/32);
//matrix_trans_shm<<<grid_size, block_size>>>(dev_A, M, N, dev_B);

void matrix_trans_cpu(int* A, int M, int N, int* B) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[j * M + i] = A[i * N + j];
        }
    }
}

__global__ void matrix_trans_shm(int* dev_A, int M, int N, int* dev_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个block处理32*32的矩阵块
    __shared__ int s_data[32][32];
  
    if (row < M && col < N) {
      // 从全局内存中加载数据，转置后写到共享内存中
      s_data[threadIdx.x][threadIdx.y] = dev_A[row * N + col];
      __syncthreads();
      int n_col = blockIdx.y * blockDim.y + threadIdx.x;
      int n_row = blockIdx.x * blockDim.x + threadIdx.y;
      if (n_col < M && n_row < N) {
        // 从转置后的共享内存按行写到全局内存结果中
        dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x];
      }
    }
}

__global__ void matrix_trans_shm_padding(int* dev_A, int M, int N, int* dev_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
  
    // 每个block处理32*32的矩阵块，尾部padding来避免bank conflict
    __shared__ int s_data[32][33];
  
    if (row < M && col < N) {
      s_data[threadIdx.x][threadIdx.y] = dev_A[row * N + col];
      __syncthreads();
      int n_col = blockIdx.y * blockDim.y + threadIdx.x;
      int n_row = blockIdx.x * blockDim.x + threadIdx.y;
      if (n_col < M && n_row < N) {
        dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x];
      }
    }
}

__global__ void matrix_trans_swizzling(int* dev_A, int M, int N, int* dev_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
  
    __shared__ int s_data[32][32];
  
    if (row < M && col < N) {
      // 从全局内存读取数据写入共享内存的逻辑坐标(row=x,col=y)
      // 其映射的物理存储位置位置(row=x,col=x^y)
      s_data[threadIdx.x][threadIdx.x ^ threadIdx.y] = dev_A[row * N + col];
      __syncthreads();
      int n_col = blockIdx.y * blockDim.y + threadIdx.x;
      int n_row = blockIdx.x * blockDim.x + threadIdx.y;
      if (n_row < N && n_col < M) {
        // 从共享内存的逻辑坐标(row=y,col=x)读取数据
        // 其映射的物理存储位置(row=y,col=x^y)
        dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x ^ threadIdx.y];
      }
    }
}

bool verify_result(int* cpu_B, int* gpu_B, int size) {
    for (int i = 0; i < size; i++) {
        if (cpu_B[i] != gpu_B[i]) {
            printf("Verification failed at index %d: CPU=%d, GPU=%d\n", 
                   i, cpu_B[i], gpu_B[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int *h_A, *h_B, *h_B_cpu;
    int *d_A, *d_B;
    
    // 分配主机内存
    h_A = (int*)malloc(M * N * sizeof(int));
    h_B = (int*)malloc(M * N * sizeof(int));
    h_B_cpu = (int*)malloc(M * N * sizeof(int));
    
    // 初始化输入数据
    for (int i = 0; i < M * N; i++) {
        h_A[i] = rand() % 100;
    }
    
    // 分配设备内存
    cudaMalloc(&d_A, M * N * sizeof(int));
    cudaMalloc(&d_B, M * N * sizeof(int));
    
    // 将数据拷贝到设备
    cudaMemcpy(d_A, h_A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    
    // 创建CUDA事件来测量时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 记录开始时间
    cudaEventRecord(start);
    
    // 启动kernel
    matrix_trans_shm<<<grid_size, block_size>>>(d_A, M, N, d_B);
    
    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 将结果拷贝回主机
    cudaMemcpy(h_B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // CPU计算参考结果
    matrix_trans_cpu(h_A, M, N, h_B_cpu);
    
    // 验证结果
    bool correct = verify_result(h_B_cpu, h_B, M * N);
    printf("Matrix transpose %s\n", correct ? "PASSED" : "FAILED");
    printf("Kernel execution time: %f ms\n", milliseconds);
    
    // 清理资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_A);
    free(h_B);
    free(h_B_cpu);
    
    return 0;
}
