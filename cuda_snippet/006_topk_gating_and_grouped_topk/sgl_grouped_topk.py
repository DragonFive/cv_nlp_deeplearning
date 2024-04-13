def biased_grouped_topk_impl(
    hidden_states: torch.Tensor,      # 输入的隐藏状态张量
    gating_output: torch.Tensor,      # 门控网络的输出，用于计算专家选择概率
    correction_bias: torch.Tensor,    # 用于修正门控输出的偏置项
    topk: int,                        # 每个token选择的专家数量
    renormalize: bool,                # 是否对选择的专家权重进行重新归一化
    num_expert_group: int = 0,        # 专家组的数量
    topk_group: int = 0,              # 每个token选择的专家组数量
):
    # 确保输入token数量匹配
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    # 对门控输出进行sigmoid激活，得到专家选择概率
    scores = gating_output.sigmoid()
    num_token = scores.shape[0]       # 获取token数量
    num_experts = scores.shape[1]     # 获取专家总数
    
    # 将scores重塑并添加修正偏置
    # 为什么 correction_bias 要 unsqueeze ： correction_bias 的形状应该是 [num_experts] ，即一个一维张量，
    # 包含每个专家的偏置值。为了能够与 scores.view(num_token, -1) 进行广播加法运算，需要将 correction_bias 扩展一个维度
    # 变成 [1, num_experts] ，这样才能与形状为 [num_token, num_experts] 的张量相加。
    # 通过 correction_bias.unsqueeze(0) 操作，在第 0 维添加了一个维度，使其形状变为 [1, num_experts] 。
    # 这样在与 scores.view(num_token, -1) 相加时，会自动广播到所有 token，相当于对每个 token 的每个专家概率都加上相应的偏置值。
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    
    # 计算每个专家组的得分：
    # 1. 将scores重塑为[num_token, num_expert_group, experts_per_group]
    # 2. 在每个组内选择top2的得分
    # 3. 对每个组的top2得分求和，得到组得分
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    
    # 选择得分最高的topk_group个专家组
    # sorted=False 参数是 PyTorch 中 torch.topk() 函数的一个选项，它表示返回的 top-k 元素不需要按照值的大小进行排序。
    # - 当 sorted=True （默认值）时，返回的 top-k 元素会按照值从大到小排序
    # - 当 sorted=False 时，返回的 top-k 元素不保证有任何特定的顺序，只保证它们是原始张量中最大的 k 个值
    # 使用 sorted=False 的好处是可以提高性能，因为排序操作需要额外的计算开销。在这个场景中，我们只关心哪些专家组被选中（即索引），而不关心它们的具体排序，
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
    
    # 创建组掩码，标记被选中的组
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    # 下面行码中，三个参数的含义如下：
    # 1. 第一个参数 1 ：表示沿着哪个维度进行散布操作。在这里， 1 表示沿着第二个维度（列方向）进行操作。
    # 2. 第二个参数 group_idx ：包含了索引信息的张量，指定了要在哪些位置填充值。
    # 3. 第三个参数 1 ：要填充的值。在这个例子中，我们用 1 来标记被选中的专家组。
    # 简单来说，这行代码的意思是：对于每个 token（行），根据 group_idx 中的索引，在 group_mask 的对应列位置填充值 1 ，表示这些专家组被选中了。
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    
    # 扩展组掩码到专家级别
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    
    # 将未选中组的专家得分设为负无穷
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]
    
    # 在选中的专家组中选择topk个专家
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    # 获取选中专家的原始得分作为权重
    topk_weights = scores.gather(1, topk_ids)

    # 如果需要重新归一化，对选中的专家权重进行归一化处理
    if renormalize:
        topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / topk_weights_sum

    # 返回归一化后的权重和选中的专家ID
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)