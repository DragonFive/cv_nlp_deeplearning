NVIDIA CUB 库可以快速实现 reduce sum

## cub DeviceReduce

```c
#include <cuda_runtime.h>
#include <stdio.h>
#include <cub/cub.cuh>    // 添加 CUB 头文件


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
```

CUB 的 DeviceReduce 方法（包括 Sum、Min、Max 等）通常需要两次执行：

1. 第一次执行：获取所需的临时存储空间大小
2. 第二次执行：实际进行规约运算

这样设计的原因：

1. CUB 会根据输入数据的大小和类型动态选择最优算法
2. 不同的算法可能需要不同大小的临时存储空间
3. 这种设计让用户可以更灵活地管理内存
优化建议：

- 如果需要多次调用同样大小的规约操作，可以复用临时存储空间
- 可以在初始化时就分配好临时存储空间，避免重复的内存分配和释放


## cub DeviceScan
是的，CUB 提供了 DeviceScan 系列方法，用于前缀和（扫描）操作。主要包括：

1. 基础扫描操作：
```cpp
DeviceScan::ExclusiveSum()    // 不包含当前元素的前缀和
DeviceScan::InclusiveSum()    // 包含当前元素的前缀和
```

举例说明区别：
- 输入数组：[1, 2, 3, 4]
- ExclusiveSum 结果：[0, 1, 3, 6]  // 每个位置是前面所有元素的和
- InclusiveSum 结果：[1, 3, 6, 10] // 每个位置是包含当前元素的前缀和

使用示例：
```cpp
// 不包含当前元素的前缀和
void *d_temp_storage = NULL;
size_t temp_storage_bytes = 0;
cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                             d_in, d_out, num_items);

// 包含当前元素的前缀和
cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                             d_in, d_out, num_items);
```

## cub 其他算法

CUDA 还有许多其他常用的并行算法模式，主要包括：

1. 排序相关：
```cpp
cub::DeviceRadixSort::SortKeys()          // 基数排序（键）
cub::DeviceRadixSort::SortPairs()         // 基数排序（键值对）
cub::DeviceMergeSort::SortKeys()          // 归并排序
```

1. 选择操作：
```cpp
cub::DeviceSelect::If()                   // 条件选择
cub::DeviceSelect::Unique()               // 去重
cub::DeviceSelect::Flagged()              // 标记选择
```

1. 直方图：
```cpp
cub::DeviceHistogram::HistogramEven()     // 等间距直方图
cub::DeviceHistogram::HistogramRange()    // 自定义范围直方图
```

1. 分区操作：
```cpp
cub::DevicePartition::If()                // 条件分区
cub::DevicePartition::Flagged()           // 标记分区
```

1. 运行长度编码：
```cpp
cub::DeviceRunLengthEncode::Encode()      // RLE编码
cub::DeviceRunLengthEncode::NonTrivial()  // 非平凡RLE
```

1. 数据重排：
```cpp
cub::DeviceSegmentedRadixSort            // 分段排序
cub::DeviceSegmentedReduce               // 分段规约
cub::DeviceSegmentedScan                 // 分段扫描
```
