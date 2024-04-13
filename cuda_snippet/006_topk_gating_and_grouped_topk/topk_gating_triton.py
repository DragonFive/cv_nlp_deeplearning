import torch
import triton
import triton.language as tl
import time

@triton.jit
def fused_topk_first_kernel(
    logits_ptr, gates_ptr, masks_ptr,
    seq_len, num_experts, k,
    BLOCK_S: tl.constexpr, EXPERTS: tl.constexpr,
):
    # 计算偏移量
    pid = tl.program_id(0)
    offs_s = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_e = tl.arange(0, EXPERTS)
    stride_s = num_experts

    # 加载logits并计算softmax
    logits_ptrs = logits_ptr + offs_s[:, None] * stride_s + offs_e[None, :]
    logits_data = tl.load(logits_ptrs, mask=offs_s < seq_len)
    logits_exp = tl.exp(logits_data)
    denom1 = tl.sum(logits_exp, axis=1)
    gates_data = logits_exp / denom1[:,None]
    
    # 存储gates结果
    gates_ptrs = gates_ptr + offs_s[:, None] * stride_s + offs_e[None, :]
    tl.store(gates_ptrs, gates_data, mask=offs_s < seq_len)

    # 计算top-k masks
    fill_value = float("-inf")
    for idx in range(k):
        max_idx = tl.argmax(gates_data, axis=1, tie_break_left=False)
        mask_data = tl.zeros((BLOCK_S, EXPERTS), tl.int64)
        all_ids = tl.broadcast_to(tl.arange(0, EXPERTS)[None,:], (BLOCK_S, EXPERTS))
        max_idx = tl.broadcast_to(tl.reshape(max_idx, (BLOCK_S, 1)), (BLOCK_S, EXPERTS))
        mask_data = tl.where(all_ids == max_idx, 1, mask_data)
        masks_ptrs = masks_ptr + idx * seq_len * EXPERTS + offs_s * stride_s + offs_e
        tl.store(masks_ptrs, mask_data, mask=offs_s < seq_len)
        gates_data = tl.where(mask_data > 0, fill_value, gates_data)

@triton.jit
def fused_topk_second_kernel(
    masks_ptr, gates_ptr, locations_ptr, res_ptr, ce_ptr,
    seq_len, num_experts, k,
    BLOCK_KS: tl.constexpr, EXPERTS: tl.constexpr,
):
    # 计算偏移量
    pid_e = tl.program_id(0)
    offs_ks = tl.arange(0, BLOCK_KS)
    stride_s = num_experts
    
    # 加载第一个mask（用于计算ce）
    masks_ptrs = masks_ptr + offs_ks * stride_s + pid_e
    mask0_data = tl.load(masks_ptrs, mask=offs_ks < seq_len)
    
    # 加载所有masks并计算locations
    masks_data = tl.load(masks_ptrs, mask=offs_ks < k)
    locations_data = tl.cumsum(masks_data, axis=0) - 1
    
    # 加载gates并计算me
    gates_ptrs = gates_ptr + offs_ks * stride_s + pid_e
    gates_data = tl.load(gates_ptrs, mask=offs_ks < seq_len)
    me = tl.sum(gates_data, axis=0) / seq_len
    
    # 计算ce和辅助损失
    ce_data = tl.sum(mask0_data, axis=0) / seq_len
    mul = me * ce_data * EXPERTS * EXPERTS
    
    # 存储结果
    res_ptrs = res_ptr + pid_e
    ce_ptrs = ce_ptr + pid_e
    locations_ptrs = locations_ptr + offs_ks * stride_s + pid_e
    
    tl.store(locations_ptrs, locations_data, mask=offs_ks < k)
    tl.store(res_ptrs, mul, mask=pid_e < EXPERTS)
    tl.store(ce_ptrs, ce_data, mask=pid_e < EXPERTS)


def fused_topkgating_triton_opt(
    logits: Tensor,
    k: int,
    capacity_factor: float,
    min_capacity: int,
    enable_token_rearrange_opt: bool = False,
    use_tutel: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """使用Triton优化的TopKGating实现"""
    batch_size, num_experts = logits.shape
    
    # 创建输出张量
    gates = torch.empty_like(logits)
    masks = torch.empty((k, batch_size, num_experts), dtype=torch.int64, device=logits.device)
    locations = torch.empty_like(masks)
    res = torch.empty(num_experts, device=logits.device)
    ce = torch.empty(num_experts, device=logits.device)
    
    # 计算grid和block大小
    BLOCK_S = 32
    grid = (triton.cdiv(batch_size, BLOCK_S),)
    
    # 调用kernel
    fused_topk_first_kernel[grid](
        logits.contiguous().data_ptr(),
        gates.contiguous().data_ptr(),
        masks.contiguous().data_ptr(),
        batch_size, num_experts, k,
        BLOCK_S=BLOCK_S, EXPERTS=num_experts,
    )
     # 第二个kernel：计算locations和辅助损失
    BLOCK_KS = 32
    grid = (num_experts,)
    fused_topk_second_kernel[grid](
        masks.contiguous().data_ptr(),
        gates.contiguous().data_ptr(),
        locations.contiguous().data_ptr(),
        res.contiguous().data_ptr(),
        ce.contiguous().data_ptr(),
        batch_size, num_experts, k,
        BLOCK_KS=BLOCK_KS, EXPERTS=num_experts,
    )
    
    
    l_aux = torch.mean(me * ce) * num_experts * num_experts
    
    # 后续处理保持不变...
    # (这里省略了与原函数相同的后续处理代码)
    
    return l_aux, top2_gating_token_infos