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

## reduce 2

当进行到最后几轮迭代时，此时的block中只有warp0在干活时，线程还在进行同步操作。这一条语句造成了极大的浪费。
由于一个warp中的32个线程其实是在一个SIMD单元上(当我们说"一个 SIMD 单元"时，实际上指的是一组协同工作的 CUDA Core，而不是单个处理核心。)，这32个线程每次都是执行同一条指令，这天然地保持了同步状态，因而当s=32时，即只有一个SIMD单元在工作时，完全可以将__syncthreads()这条同步代码去掉。所以我们将最后一维进行展开以减少同步。

为什么warp中的线程不需要同步
- SIMT机制：在CUDA中，warp中的32个线程执行相同的指令，但每个线程处理不同的数据。由于它们执行相同的指令，因此在硬件层面不需要额外的同步机制。硬件会自动确保这些线程在执行相同指令时的同步性。
- 锁步执行：warp中的线程以**锁步（lock-step）**的方式执行指令。这意味着在每个时钟周期内，所有线程都会执行相同的指令，硬件会自动协调这些线程的执行，确保它们在同一时间点执行相同的操作。
- 硬件调度：CUDA的硬件调度器（Warp Scheduler）会负责管理warp的执行。它会确保warp中的线程在执行时保持同步，而不需要程序员显式地进行同步操作

SIMT (Single Instruction Multiple Thread) 是 NVIDIA GPU 的核心执行模型，主要特点如下：

1. 基本概念：
   
   - 单指令：所有线程执行相同的指令
   - 多线程：每个线程处理不同的数据
   - 类似于 SIMD（Single Instruction Multiple Data），但更灵活
2. 执行特点：
   
   - 以 warp 为基本执行单位（通常是32个线程）
   - warp 中的线程同时执行相同的指令
   - 每个线程有自己的程序计数器和寄存器状态
   - 支持线程分支，但可能影响性能

```c++
__device__ void warpReduce(volatile float* cache,int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}

__global__ void reduce4(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid<32)warpReduce(sdata,tid);
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}
```

对于GPU而言，block的取值到底是多更好，还是少更好。有的同学，肯定会说：“那肯定是多更好啦。Block数量多，block可以进行快速地切换，去掩盖访存的延时。”

如果一个线程被分配更多的work时，可能会更好地覆盖延时。这一点比较好理解。如果线程有更多的work时，对于编译器而言，就可能有更多的机会对相关指令进行重排，从而去覆盖访存时的巨大延时。在某种程度上而言，block少一些会更好,block需要进行合理地设置。

## shuffle

NV出了Shuffle指令，对于reduce优化有着非常好的效果。目前绝大多数访存类算子，像是softmax，batch_norm，reduce等，都是用Shuffle实现。
Shuffle指令是一组针对warp的指令。Shuffle指令最重要的特性就是warp内的寄存器可以相互访问。在没有shuffle指令的时候，各个线程在进行通信时只能通过shared memory来访问彼此的寄存器。而采用了shuffle指令之后，**warp内的线程可以直接对其他线程的寄存器进行访存**。通过这种方式可以减少访存的延时。除此之外，带来的最大好处就是可编程性提高了，在某些场景下，就不用shared memory了。

使用 shuffle 指令的主要优势：
1. 减少共享内存访问
2. 降低线程同步开销
3. 提高 warp 内数据交换效率
4. 减少内存带宽压力

warp shuffle 适合二维block结构