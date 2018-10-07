




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

