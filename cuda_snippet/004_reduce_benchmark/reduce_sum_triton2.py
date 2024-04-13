import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'NUM_PER_THREAD': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_PER_THREAD': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_PER_THREAD': 2}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_PER_THREAD': 1}, num_warps=32),
    ],
    key=['n_elements'],
)
@triton.jit
def reduce_sum_kernel(
    x_ptr,  
    output_ptr,  
    n_elements,  
    BLOCK_SIZE: tl.constexpr,
    NUM_PER_THREAD: tl.constexpr,  # 每个线程处理的元素数量
):
    pid = tl.program_id(0)
    
    # 计算当前block需要处理的数据范围
    block_start = pid * (BLOCK_SIZE * NUM_PER_THREAD)
    
    # 使用 multiple_of 提示编译器数据对齐
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    offsets = tl.multiple_of(offsets, 8)
    
    # 初始化累加值
    sum_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # 每个线程处理多个元素
    for i in range(NUM_PER_THREAD):
        curr_offsets = offsets + i * BLOCK_SIZE
        mask = curr_offsets < n_elements
        x = tl.load(x_ptr + curr_offsets, mask=mask, other=0.0)
        sum_acc += x.to(tl.float32)
    
    # 执行reduce sum操作，使用 float32 提高精度
    block_sum = tl.sum(sum_acc, axis=0)
    
    # 将结果写入输出数组
    tl.store(output_ptr + pid, block_sum)

def reduce_sum(x):
    # 确保输入是contiguous的
    x = x.contiguous()
    
    # 计算grid大小，考虑每个线程处理多个元素
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)  # 让编译器自动选择最佳配置
    
    # 分配输出内存并确保对齐
    output = torch.empty(grid[0], device=x.device, dtype=torch.float32)
    
    # 启动kernel
    reduce_sum_kernel[grid](
        x, output,
        n_elements,
    )
    
    if output.numel() > 1:
        return reduce_sum(output)
    return output[0]

def main():
    # 创建与CUDA版本相同大小的输入数据
    N = 32 * 1024 * 1024
    x = torch.arange(N, dtype=torch.float32, device='cuda') % 456
    
    # 预热
    for _ in range(10):
        result = reduce_sum(x)
    
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


# reduce3 0.190ms
# reduce_sum_trition.py 0.476ms
# reduce_sum_triton2.py 0.176ms

# TRITON_PRINT_AUTOTUNING=1 python3 reduce_sum_triton2.py
# Triton autotuning for function reduce_sum_kernel finished after 2.53s; best config selected: BLOCK_SIZE: 512, NUM_PER_THREAD: 2, num_warps: 16, num_ctas: 1, num_stages: 2, maxnreg: None;
# Triton autotuning for function reduce_sum_kernel finished after 0.42s; best config selected: BLOCK_SIZE: 128, NUM_PER_THREAD: 8, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None;
# Triton autotuning for function reduce_sum_kernel finished after 0.42s; best config selected: BLOCK_SIZE: 256, NUM_PER_THREAD: 4, num_warps: 8, num_ctas: 1, num_stages: 2, maxnreg: None;
# Triton reduce sum cost time: 0.176 ms
# Result correct: True
# Result: 7616835072.0
# Expected: 7616835072.0