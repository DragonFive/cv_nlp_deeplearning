import torch
import triton
import triton.language as tl
import time

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}),
    ],
    key=['M', 'N'],
)
@triton.jit
def transpose_kernel(
    output_ptr, input_ptr,
    M, N,
    stride_out_m, stride_out_n,
    stride_in_m, stride_in_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # 计算程序块索引
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # 计算偏移量
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # 创建mask处理边界情况
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # 加载输入数据
    x = tl.load(input_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n, mask=mask)
    
    # 转置并写回
    output = tl.trans(x)
    tl.store(output_ptr + offs_n[:, None] * stride_out_m + offs_m[None, :] * stride_out_n, output, mask=mask.T)

def transpose(x):
    M, N = x.shape
    output = torch.empty((N, M), device=x.device, dtype=x.dtype)
    
    # 计算stride
    stride_in_m, stride_in_n = x.stride()
    stride_out_m, stride_out_n = output.stride()
    
    # 启动kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    transpose_kernel[grid](
        output, x,
        M, N,
        stride_out_m, stride_out_n,
        stride_in_m, stride_in_n,
    )
    return output

def main():
    # 设置问题规模
    M, N = 1024, 2048
    
    # 创建输入数据
    x = torch.randint(0, 100, (M, N), device='cuda', dtype=torch.int32)
    
    # 预热
    for _ in range(10):
        transpose(x)
    torch.cuda.synchronize()
    
    # 测试Triton实现
    start_time = time.time()
    y_triton = transpose(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) * 1000  # 转换为毫秒
    
    # 测试PyTorch实现作为参考
    start_time = time.time()
    y_torch = x.T.contiguous()
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) * 1000  # 转换为毫秒
    
    # 验证结果
    print(f"正确性验证: {torch.allclose(y_triton, y_torch)}")
    print(f"Triton 执行时间: {triton_time:.3f} ms")
    print(f"PyTorch 执行时间: {torch_time:.3f} ms")
    print(f"加速比: {torch_time/triton_time:.2f}x")

if __name__ == "__main__":
    main()

