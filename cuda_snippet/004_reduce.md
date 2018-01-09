nv 有个 reduce 优化的pdf，[reduce](https://developer.download.nvidia.cn/assets/cuda/files/reduction.pdf)

在GPU中，reduce采用了一种树形的计算方式。从上至下，将数据不断地累加，直到得出最后的结果。由于GPU没有针对global数据的同步操作，只能针对block的数据进行同步。

首先需要将数组分为m个小份。而后，在第一阶段中，开启m个block计算出m个小份的reduce值。最后，在第二阶段中，使用一个block将m个小份再次进行reduce，得到最终的结果。

```c
__global__ void reduce(T *input, T* output)
```

## baseline

```c
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
```

这个代码里会有warp divergence的问题，share memory 也会bank conflict。


