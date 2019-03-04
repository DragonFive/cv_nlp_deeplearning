
## 指令发射
同一个block的warp只能在同一个SM上运行，但是同一SM可以可以容纳来自不同block甚至不同grid的若干个warp。
指令发射和warp调度的问题，就是指同一个SM内同一个warp或是不同warp的指令之间是按照什么逻辑来调度运行的。

指令发射的一些基本逻辑：

- 每个指令都需要有对应的功能单元（Functional Unit）来执行。比如执行整数指令的单元，执行浮点运算指令的浮点单元，执行跳转的分支单元等等。功能单元的个数决定了这种指令的极限发射带宽。
- 每个指令都要dispatch unit经由dispatch port进行发射。不同的功能单元可能会共用dispatch port，这就意味着这些功能单元的指令需要通过竞争来获得发射机会。不同的架构dispatch port的数目和与功能单元分配情况会有一些差别。
- 有些指令由于功能单元少，需要经由同一个dispatch port发射多次，这样dispatch port是一直占着的，期间也不能发射其他指令。比较典型的是F64指令。
- 每个指令能否发射还要满足相应的依赖关系和资源需求。比如指令LDG.E R6, [R2] ;首先需要等待之前写入R[2:3]的指令完成，其次需要当前memory IO的queue还有空位，否则指令也无法下发。在指令要下发到memory IO进行处理时，需要先查看memory IO的队列（queue），只有当这个队列中还有空余位置（空位）时，指令才能顺利进入队列并被下发。如果队列已满，那么指令就无法下发，需要等待队列中有空位出现。

### kelper 架构
Kepler的GTX 780 Ti有15个SMX，SMX有192个core，4个warp scheduler， 8个 dispatch unit。每个 warp scheduler每cycle可以选中一个warp，每个dispatch unit每cycle可以issue一个指令。
由于core多且与warp没有对应关系，kepler的dual issue不一定是发给两个不同的功能单元，两个整数、两个F32或是混合之类的搭配应该也是可以的，关键是要有足够多的空闲core。每个warp scheduler配了两个dispatch unit，共8个，而发射带宽填满cuda core只要6个就够了，多出来的2个可以用来双发射一些load/store或是branch之类的指令。

### Maxwell 架构
Maxwell 的每个SM有32个core，有2个warp scheduler，每个warp scheduler有2个dispatch unit。每个warp scheduler配了两个dispatch unit，共4个，而发射带宽填满cuda core只要2个就够了，多出来的2个可以用来双发射一些load/store或是branch之类的指令。

### Turing 架构
Turing的SM的core数减半，变成64个，但还是分成4个区，每个区16个core，配一个warp scheduler和一个dispatch unit。这样两个cycle发一个指令就足够填满所有core了，另一个cycle就可以用来发射别的指令。
从Volta开始，NV把整数和一些浮点数的pipe分开，使用不同的dispatch port。这样，一些整数指令和浮点数指令就可以不用竞争发射机会了。一般整数指令用途很广，即使是浮点运算为主的程序，也仍然需要整数指令进行一些地址和辅助运算。因此，把这两者分开对性能还是很有一些帮助的。




- ## F64 指令
在 CUDA 中，F64 指令通常指的是与双精度浮点数（double 类型，占用 64 位）相关的指令。

```c
__global__ void f64_operations(double *a, double *b, double *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double x = a[idx];
    double y = b[idx];

    double sum = x + y; // add.f64
    double diff = x - y; // sub.f64
    double prod = x * y; // mul.f64
    double fma_result = fma(x, y, x); // fma.f64

    result[idx * 4] = sum;
    result[idx * 4 + 1] = diff;
    result[idx * 4 + 2] = prod;
    result[idx * 4 + 3] = fma_result;
}
```

- 加法：

```c
add.f64 %d, %a, %b;
```
该指令将两个双精度浮点数 %a 和 %b 相加，结果存储在 %d 中。


- 融合乘加（FMA）

```c
fma.rnd.f64 %d, %a, %b, %c;
```
该指令执行融合乘加操作，计算 %a * %b + %c，并将结果存储在 %d 中。

  - .rnd 表示四舍五入模式，可选值包括 
  - .rn（四舍五入到最近值）
  - .rz（向零舍入）
  - .rm（向负无穷舍入）
  - .rp（向正无穷舍入）

关于fma有以下注意点：
- fma() 函数是内置的，包含在 <cuda_runtime.h> 中，通常这个头文件在包含 <cuda.h> 时就会自动包含。
- fma() 函数是重载的，它会根据传入参数的类型自动选择对应的精度版本。
- 在CUDA C/C++层面， fma() 函数使用的是默认的舍入模式（通常是.rn，即四舍五入到最近值）
- 如果需要指定不同的舍入模式，你需要使用PTX汇编级别的指令，或者使用CUDA数学函数库提供的特殊函数

如果需要明确指定类型和舍入模式，需要使用PTX内联汇编
```c
    double fma_result_rz;
    asm("fma.rz.f64 %0, %1, %2, %3;" : "=d"(fma_result_rz) : "d"(x), "d"(y), "d"(x));
```

## LDG 与 LDS

LDG.E 是CUDA PTX 汇编中的一个加载（Load）指令。
LDG.E R6, [R2] 的含义是：

- LDG ：Load Global Memory，从全局内存中加载数据
- .E ：表示"经过缓存的"（cached）访问方式
- R6 ：目标寄存器，数据将被加载到这个寄存器中
- [R2] ：源地址，R2寄存器中存储的是全局内存的地址

其他相关的变体包括：
- LDG.U ：未缓存的全局内存加载
- LDG.CG ：通过常量缓存加载
- LDG.CS ：通过共享内存加载

LDS ：Load from Shared Memory，从共享内存加载数据到寄存器，在 CUDA kernel 中访问 __shared__ 声明的共享内存变量时使用。
LDS.128 是CUDA PTX中的一个特殊的共享内存加载指令，用于一次性加载128位（16字节）的数据。
1. 一次加载128位数据（相当于4个32位或2个64位数据）
2. 通常用于向量化加载操作
3. 要求内存地址必须是16字节对齐的

