import torch
import triton
import triton.language as tl
import time

def fused_topkgating_opt(
    logits: Tensor,
    k: int,
    capacity_factor: float,
    min_capacity: int,
    enable_token_rearrange_opt: bool = False,
    use_tutel: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements TopKGating on logits."""
# first section————————————————————————————————————————————————————————————————————
    gates = F.softmax(logits, dim=1)
    num_experts = int(gates.shape[1])
    capacity = _capacity(gates, torch.tensor(capacity_factor * k), torch.tensor(min_capacity))
    # 获取top-k的值和索引, 
    # - 在专家维度上选择概率最高的k个专家
    # - .indices 获取这k个专家的索引
    # - .t() 进行转置操作, 为什么要转置?
    # 便于后续的reshape(-1)操作，转置后展平的结果会按照"每个token的第1个选择、每个token的第2个选择..."这样的顺序排列
    indices_s = torch.topk(gates, k, dim=1).indices.t()
    # - 将indices_s展平后转换为one-hot编码
    # - num_classes=num_experts指定one-hot编码的类别数
    # - 输出的masks标记了每个token选择的专家
    masks = F.one_hot(indices_s.reshape(-1), num_classes=num_experts)

# second section————————————————————————————————————————————————————————————————————
    if use_tutel and TUTEL_INSTALLED:
        locations = tutel_moe.fast_cumsum_sub_one(masks)
    else:
        # 这行代码计算每个专家（expert）的累积和，并减1
        # 目的是为每个被选中的token分配在对应专家队列中的位置
        # 例如，如果某个专家被选中了3次，那么这3个token会被分配到位置0,1,2
        locations = torch.cumsum(masks, dim=0) - 1
    # 将masks和locations重塑为三维张量
    # 维度分别是：[k（top-k）, batch_size, num_experts]
    # 这样便于后续的并行计算，重塑之前shape为[k * batch_size, num_experts]
    masks = masks.reshape(-1, gates.shape[0], num_experts)
    locations = locations.reshape(-1, gates.shape[0], num_experts)
    # 计算gates在batch维度上的平均值表示每个专家被选中的平均概率
    me = torch.mean(gates, dim=0)
    # - 计算masks在batch维度上的平均值，表示每个专家实际被选中的比例
    ce = torch.mean(masks[0].type_as(logits), dim=0)
    # - 这是计算辅助损失（auxiliary loss），目的是为了平衡专家的使用，防止某些专家被过度使用而其他专家闲置
    l_aux = torch.mean(me * ce) * num_experts * num_experts
   
# third section————————————————————————————————————————————————————————————————————
    # 将超出容量的专家位置置为0
    masks *= torch.lt(locations, capacity)
    # 两者相乘，去掉中间用来传递的其实没有被选中的token。
    locations_s = torch.sum(locations * masks, dim=2)
    # - 将masks转换为与logits相同的数据类型（通常是float）
    # - 为后续的浮点数计算做准备
    mask_float = masks.type_as(logits)
    # 这个乘法的目的是：
    # - 将未被选中的专家（mask中为0的位置）对应的概率置为0
    # - 只保留每个token在其top-k专家中的概率值
    gate_s, indices_s = torch.max(gates * mask_float, dim=2)
    # 计算每个样本的gate值总和
    # 用于后续的归一化
    denom_s = torch.sum(gate_s, dim=0)
    # 对denom_s进行限制，确保其最小值为eps（一个非常小的正数）
    # 这是为了防止在归一化过程中分母为0
    # torch.clamp 是一个用于限制张量中元素值范围的函数。它有三个主要参数：
    # - input：输入张量
    # - min：最小值界限
    # - max：最大值界限（这里没有使用）
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    # 进行归一化
    gate_s /= denom_s
    # 算每个token在所有专家队列中的全局位置
    token_rearranged_ec_idx = indices_s.int() * capacity + locations_s.int()
    # - 为不同优先级的专家选择赋予不同的权重
    # - torch.arange(k, 0, -1) 创建从k到1的递减序列
    # - 通过乘法操作将权重应用到masks上
    token_sel_exp_int_mask = masks * torch.arange(k, 0, -1, device=masks.device).reshape(k, 1, 1)
        
#forth section————————————————————————————————————————————————————————————————————
    """
    1. token_sel_exp_int_mask 的shape是 [k, batch_size, num_experts]
    
    - k: top-k中的k
    - batch_size: 批次大小
    - num_experts: 专家数量
    2. torch.sum(token_sel_exp_int_mask, dim=0) 在第0维（k维度）上求和：
    
    - 消除了k这个维度
    - 输出shape变为 [batch_size, num_experts]
    这个求和操作的目的是：

    - 将每个token对不同专家的优先级权重（从k到1的递减序列）累加起来
    - 得到每个token对每个专家的总体优先级分数
    - 这个分数后续用于选择每个专家要处理的tokens
    """
    # 用 torch.topk 选择每个专家要处理的前capacity个最高优先级的tokens
    expert_sel_top_c_token_idx = torch.topk(
         torch.sum(token_sel_exp_int_mask, dim=0), k=capacity, dim=0, sorted=True
        )[1]
    expert_select_token_idx = expert_sel_top_c_token_idx.t().reshape(num_experts * capacity)
    token_rearranged_ec_idx = token_rearranged_ec_idx.reshape(-1)
    token_exp_weights = gate_s.reshape(-1)
    
    top2_gating_token_infos = GatingTokenRearrangeInfo(
        token_rearranged_ec_idx=token_rearranged_ec_idx,
        token_exp_weights=token_exp_weights,
        expert_select_token_idx=expert_select_token_idx,
        )
    return l_aux, top2_gating_token_infos

