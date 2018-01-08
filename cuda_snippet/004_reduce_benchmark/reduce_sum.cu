#include <cuda_runtime.h>
#include <stdio.h>

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

// CPU版本的reduce sum用于验证结果
float cpu_sum(float *arr, int n) {
    float sum = 0;
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
    
    // CPU版本计算结果用于验证
    float cpu_result = cpu_sum(h_in, NUM_ELEMENTS);
    
    // 输出结果
    printf("GPU计算结果: %.0f\n", gpu_sum);
    printf("CPU计算结果: %.0f\n", cpu_result);
    printf("Kernel执行时间: %.3f ms\n", milliseconds);
    printf("带宽: %.2f GB/s\n", 
           (NUM_ELEMENTS * sizeof(float)) / (milliseconds * 1000000));
    
    // 清理
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_in);
    free(h_out);
    
    return 0;
}
