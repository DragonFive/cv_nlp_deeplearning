#include <cuda_runtime.h>
#include <stdio.h>
#include <cub/cub.cuh> 

#define THREAD_PER_BLOCK 256
#define NUM_ELEMENTS (32 * 1024 * 1024)  // 32M elements

// 原有的 reduce kernel 保持不变
__global__ void reduce0(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s<blockDim.x; s*=2){
        if(tid%(2*s) == 0){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}

__global__ void reduce2(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x>>1; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

// 使用 CUB 实现的 reduce sum
void cub_reduce_sum(float *d_in, float *d_out, int num_elements) {
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    
    // 第一次调用获取临时存储大小
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, 
                          d_in, d_out, num_elements);
    
    // 分配临时存储
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // 执行规约操作
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, 
                          d_in, d_out, num_elements);
                          
    // 清理
    cudaFree(d_temp_storage);
}

// CPU版本的reduce sum用于验证结果
double cpu_sum(float *arr, int n) {
    double sum = 0;
    for(int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    float *h_in, *h_out;    // host数据
    float *d_in, *d_out;    // device数据
    
    // 分配主机内存
    h_in = (float*)malloc(NUM_ELEMENTS * sizeof(float));
    h_out = (float*)malloc((NUM_ELEMENTS/THREAD_PER_BLOCK) * sizeof(float));
    
    // 初始化输入数据
    for(int i = 0; i < NUM_ELEMENTS; i++) {
        h_in[i] = 1.0f;  // 全部填充1，方便验证
    }
    
    // 分配设备内存
    cudaMalloc(&d_in, NUM_ELEMENTS * sizeof(float));
    cudaMalloc(&d_out, (NUM_ELEMENTS/THREAD_PER_BLOCK) * sizeof(float));
    
    // 将数据拷贝到设备
    cudaMemcpy(d_in, h_in, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    
    // 配置kernel启动参数
    int num_blocks = NUM_ELEMENTS / THREAD_PER_BLOCK;
    dim3 grid(num_blocks, 1, 1);
    dim3 block(THREAD_PER_BLOCK, 1, 1);
    
    // 创建CUDA事件来测量时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 开始计时
    cudaEventRecord(start);
    
    // 启动kernel
    reduce0<<<grid, block>>>(d_in, d_out);
    
    // 结束计时
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 计算经过的时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 将结果拷贝回主机
    cudaMemcpy(h_out, d_out, (NUM_ELEMENTS/THREAD_PER_BLOCK) * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 计算最终结果（需要对block的结果再求和）
    float gpu_sum = 0;
    for(int i = 0; i < num_blocks; i++) {
        gpu_sum += h_out[i];
    }
    // 测试 CUB 实现
    float *d_cub_out;
    cudaMalloc(&d_cub_out, sizeof(float));
    
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    
    cudaEventRecord(start2);
    cub_reduce_sum(d_in, d_cub_out, NUM_ELEMENTS);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start2, stop2);
    
    // 获取 CUB 结果
    float gpu_sum2;
    cudaMemcpy(&gpu_sum2, d_cub_out, sizeof(float), cudaMemcpyDeviceToHost);
    // CPU版本计算结果用于验证
    // 测量 CPU 计算时间
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    
    cudaEventRecord(start3);
    float cpu_result = float(cpu_sum(h_in, NUM_ELEMENTS));
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    
    float milliseconds3 = 0;
    cudaEventElapsedTime(&milliseconds3, start3, stop3);
    // 输出结果
    printf("reduce0 kernel计算结果: %.0f\n", gpu_sum);
    printf("CUB实现结果: %.0f\n", gpu_sum2);
    printf("CPU计算结果: %.0f\n", cpu_result);
    printf("reduce0 Kernel执行时间: %.3f ms\n", milliseconds);
    printf("CUB实现时间: %.3f ms\n", milliseconds2);
    printf("CPU执行时间: %.3f ms\n", milliseconds3);
    printf("reduce0 kernel带宽: %.2f GB/s\n", 
           (NUM_ELEMENTS * sizeof(float)) / (milliseconds * 1000000));
    printf("CUB实现带宽: %.2f GB/s\n", 
            (NUM_ELEMENTS * sizeof(float)) / (milliseconds2 * 1000000));
    
    // 清理
    // 清理额外的资源
    cudaFree(d_cub_out);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_in);
    free(h_out);
    
    return 0;
}
