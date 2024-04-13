import torch
import triton
import triton.language as tl

@triton.jit
def reduce_sum_kernel(
    x_ptr,  # 指向输入数据的指针
    output_ptr,  # 指向输出数据的指针
    n_elements,  # 输入数组的总元素数
    BLOCK_SIZE: tl.constexpr,  # 每个block处理的元素数量
):
    # 计算当前block的索引
    pid = tl.program_id(0)
    
    # 计算当前block需要处理的数据范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建mask来处理边界情况
    mask = offsets < n_elements
    
    # 从全局内存加载数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 执行reduce sum操作
    block_sum = tl.sum(x, axis=0)
    
    # 将结果写入输出数组
    tl.store(output_ptr + pid, block_sum)

def reduce_sum(x):
    # 确保输入是contiguous的
    x = x.contiguous()
    
    # 定义block大小和grid大小
    BLOCK_SIZE = 1024
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # 分配输出内存
    output = torch.empty(grid[0], device=x.device, dtype=x.dtype)
    
    # 启动kernel
    reduce_sum_kernel[grid](
        x, output,
        n_elements,
        BLOCK_SIZE,
    )
    
    # 如果需要进一步reduce，递归调用
    if output.numel() > 1:
        return reduce_sum(output)
    return output[0]

# 测试代码
def main():
    # 创建与CUDA版本相同大小的输入数据
    N = 32 * 1024 * 1024
    x = torch.arange(N, dtype=torch.float32, device='cuda') % 456
    
    # 计时
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(2000):
        result = reduce_sum(x)
    end.record()
    
    torch.cuda.synchronize()
    print(f"Triton reduce sum cost time: {start.elapsed_time(end) / 2000:.3f} ms")
    
    # 验证结果
    expected = x.sum()
    print(f"Result correct: {torch.allclose(result, expected)}")
    print(f"Result: {result}")
    print(f"Expected: {expected}")

if __name__ == "__main__":
    main()